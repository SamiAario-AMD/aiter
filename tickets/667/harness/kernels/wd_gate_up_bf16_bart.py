"""FlyDSL warp-decode gate_up kernel -- Task 0 (BF16 act x BF16 weights).

One wave (64 lanes) per output scalar inter[token_b, expert_k, neuron_j].

Grid: (B*TOPK*INTER,)  Block: (64,)

Two codepaths selected by kUseDot2:
  kUseDot2=False (default, gfx942-safe): convert BF16 to FP32 via bit-shift,
    accumulate with scalar v_mac_f32 / v_fma_f32.  Correct on all CDNA.
  kUseDot2=True  (gfx950 only): v_dot2_f32_bf16 via inline asm.  Use this
    on the gfx950 node for the performance comparison.

API: compile_wd_moe_gate_up(...) returns a @flyc.jit launch wrapper.
Call it with (_ptr(inter_out), _ptr(x), _ptr(w_gate), _ptr(w_up),
              _ptr(router_ids), B, topk, inter, hidden, experts, stream).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, buffer_ops, rocdl, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import ArithValue
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl._mlir.dialects.arith import CmpIPredicate

WAVE_SIZE = 64


def _bf16_to_f32(bf16_bits_i32, which):
    """Unpack one BF16 from a packed i32 word and widen to FP32.

    BF16 is stored in the upper 16 bits of FP32 (same exponent/mantissa fields,
    just 7 fewer mantissa bits).  So bf16->f32 = left-shift raw bits by 16.

    which=0: low  16 bits (bits [15:0])
    which=1: high 16 bits (bits [31:16])
    """
    i32 = T.i32
    f32 = T.f32
    if which == 0:
        # Mask low 16, shift left 16 to place in upper half of f32
        lo = arith.andi(bf16_bits_i32, arith.constant(0xFFFF, type=i32))
        shifted = arith.shli(lo, arith.constant(16, type=i32))
    else:
        # High 16 bits already in position -- mask and zero low 16
        shifted = arith.andi(bf16_bits_i32, arith.constant(0xFFFF0000, type=i32))
    return arith.bitcast(f32, shifted)


def _dot2_f32_bf16_inline(acc_f32, a_i32, b_i32):
    """v_dot2_f32_bf16 + s_nop 2 (gfx950 only, dependent-chain form).

    Each i32 carries two packed BF16 elements.
    dst += a[0]*b[0] + a[1]*b[1]  (FP32 accumulate, 2 MACs/lane/cycle).
    """
    result = llvm.inline_asm(
        T.f32,
        [acc_f32, a_i32, b_i32],
        "v_dot2_f32_bf16 $0, $2, $3, $1",
        "=v,v,v,v",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    rocdl.s_nop(2)
    return result


def _butterfly_reduce(val_f32):
    """6-step XOR butterfly: all 64 lanes end up with the total sum."""
    av = ArithValue(val_f32)
    for stage in range(6):
        offset = 1 << stage
        peer = av.shuffle_xor(offset, WAVE_SIZE)
        av = ArithValue(arith.addf(av, ArithValue(peer)))
    return av  # ArithValue is an ir.Value subclass


@functools.lru_cache(maxsize=None)
def compile_wd_moe_gate_up(
    *,
    hidden: int,
    inter: int,
    experts: int,
    topk: int,
    kVector: int = 8,
    a_dtype: str = "bf16",
    w_dtype: str = "bf16",
    use_dot2: bool = False,
):
    """Compile warp-decode gate_up, return @flyc.jit launch wrapper.

    use_dot2=False (default): FP32 scalar path, runs on gfx942 and gfx950.
    use_dot2=True: v_dot2_f32_bf16 inline asm, gfx950 only.
    """
    if a_dtype != "bf16" or w_dtype != "bf16":
        raise NotImplementedError(
            f"Only bf16xbf16 implemented (Task 0). Got a={a_dtype!r}, w={w_dtype!r}"
        )

    assert (
        hidden % (WAVE_SIZE * kVector) == 0
    ), f"hidden={hidden} must be divisible by {WAVE_SIZE * kVector}"

    dot2_tag = "_dot2" if use_dot2 else "_f32"
    module_name = (
        f"wd_gate_up_a{a_dtype}_w{w_dtype}"
        f"_h{hidden}_i{inter}_e{experts}_topk{topk}_kv{kVector}{dot2_tag}"
    ).replace("-", "_")

    _kPairs = kVector // 2
    _kStep = WAVE_SIZE * kVector
    _num_k_steps = hidden // _kStep

    @flyc.kernel(name=module_name, known_block_size=[WAVE_SIZE, 1, 1])
    def _kernel(
        arg_inter_out: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w_gate: fx.Pointer,
        arg_w_up: fx.Pointer,
        arg_router_ids: fx.Pointer,
        i32_B: fx.Int32,
        i32_TOPK: fx.Int32,
        i32_INTER: fx.Int32,
        i32_HIDDEN: fx.Int32,
        i32_E: fx.Int32,
    ):
        f32 = T.f32
        i32 = T.i32
        bf16 = T.bf16

        B_i32 = i32_B.ir_value()
        TOPK_i32 = i32_TOPK.ir_value()
        INTER_i32 = i32_INTER.ir_value()
        HIDDEN_i32 = i32_HIDDEN.ir_value()

        # -- Buffer resources -------------------------------------------------
        def rsrc(ptr, nbytes_i32):
            addr64 = arith.index_cast(T.i64, fx.ptrtoint(ptr))
            return buffer_ops.create_buffer_resource_from_addr(
                addr64, num_records_bytes=nbytes_i32
            )

        max_slots = B_i32 * TOPK_i32
        x_nb = max_slots * HIDDEN_i32 * arith.constant(2, type=i32)
        x_rsrc = rsrc(arg_x, x_nb)
        rid_nb = max_slots * arith.constant(4, type=i32)
        rid_rsrc = rsrc(arg_router_ids, rid_nb)
        out_nb = max_slots * INTER_i32 * arith.constant(2, type=i32)
        out_rsrc = rsrc(arg_inter_out, out_nb)
        # Weight resources are created per-row below to avoid i32 overflow.

        # -- Thread / block decoding (all i32) --------------------------------
        lane_i32 = arith.index_cast(i32, gpu.thread_id("x"))
        blk_i32 = arith.index_cast(i32, gpu.block_id("x"))

        neuron_j = arith.remui(blk_i32, INTER_i32)
        blk_div = arith.divui(blk_i32, INTER_i32)
        expert_k = arith.remui(blk_div, TOPK_i32)
        token_b = arith.divui(blk_div, TOPK_i32)

        rid_off = token_b * TOPK_i32 + expert_k
        expert_e = buffer_ops.buffer_load(rid_rsrc, rid_off, vec_width=1, dtype=i32)

        # -- Per-row buffer resources (avoids i32 overflow on large matrices) --
        # Weight row byte address computed in i64 to avoid overflow.
        # For deepseek-v3: expert=255, neuron=2047, hidden=7168 ->
        #   row_elem = (255*2048+2047)*7168 = 3.76 billion > INT32_MAX.
        # Solution: bump the base pointer to the start of this row (i64 arithmetic),
        # then use a small i32 lane offset (max = hidden*2 = 14336 bytes) within the row.
        i64 = T.i64
        c_two_i64 = arith.constant(2, type=i64)
        HIDDEN_i64 = arith.extsi(i64, HIDDEN_i32)
        INTER_i64 = arith.extsi(i64, INTER_i32)
        expert_i64 = arith.extsi(i64, expert_e)
        neuron_i64 = arith.extsi(i64, neuron_j)

        # row byte offset = (expert_e * INTER + neuron_j) * HIDDEN * 2  (BF16)
        w_row_byte_off = (expert_i64 * INTER_i64 + neuron_i64) * HIDDEN_i64 * c_two_i64

        wg_base_i64 = arith.index_cast(i64, fx.ptrtoint(arg_w_gate))
        wu_base_i64 = arith.index_cast(i64, fx.ptrtoint(arg_w_up))
        wg_row_addr = arith.addi(wg_base_i64, w_row_byte_off)
        wu_row_addr = arith.addi(wu_base_i64, w_row_byte_off)

        # row_nb = hidden * 2 bytes (fits in i32; max = 14336 for deepseek)
        row_nb = HIDDEN_i32 * arith.constant(2, type=i32)
        wg_row_rsrc = buffer_ops.create_buffer_resource_from_addr(
            wg_row_addr, num_records_bytes=row_nb
        )
        wu_row_rsrc = buffer_ops.create_buffer_resource_from_addr(
            wu_row_addr, num_records_bytes=row_nb
        )

        # x row: token_b * HIDDEN * 2 bytes (token_b fits in i32 for decode)
        x_row_base = token_b * HIDDEN_i32  # i32 BF16 element offset

        # -- K-loop -----------------------------------------------------------
        lane_kV = lane_i32 * arith.constant(kVector, type=i32)
        c_kStep = arith.constant(_kStep, type=i32)
        c_two = arith.constant(2, type=i32)
        zero_f32 = arith.constant(0.0, type=f32)

        c0_idx = arith.constant(0, index=True)
        c1_idx = arith.constant(1, index=True)
        n_steps = arith.constant(_num_k_steps, index=True)

        for_op = scf.ForOp(c0_idx, n_steps, c1_idx, iter_args=[zero_f32, zero_f32])
        with ir.InsertionPoint(for_op.body):
            step_i32 = arith.index_cast(i32, for_op.induction_variable)
            g_acc = for_op.inner_iter_args[0]
            u_acc = for_op.inner_iter_args[1]

            k_base = step_i32 * c_kStep
            lane_k = k_base + lane_kV

            g_cur = g_acc
            u_cur = u_acc

            for p in range_constexpr(_kPairs):
                p2 = arith.constant(p * 2, type=i32)
                # i32 element offset within the row (max = hidden/2 words, fits i32)
                w_off = (lane_k + p2) // c_two
                # i32 word index into x (x_row_base + lane_k + p2*0 : / 2)
                x_off = (x_row_base + lane_k + p2) // c_two

                # Load i32 words (each = 2 packed BF16)
                x_word = buffer_ops.buffer_load(x_rsrc, x_off, vec_width=1, dtype=i32)
                g_word = buffer_ops.buffer_load(
                    wg_row_rsrc, w_off, vec_width=1, dtype=i32
                )
                u_word = buffer_ops.buffer_load(
                    wu_row_rsrc, w_off, vec_width=1, dtype=i32
                )

                if use_dot2:
                    # gfx950-only: v_dot2_f32_bf16 (2 MACs/lane/cycle)
                    g_cur = _dot2_f32_bf16_inline(g_cur, x_word, g_word)
                    u_cur = _dot2_f32_bf16_inline(u_cur, x_word, u_word)
                else:
                    # gfx942-safe: unpack BF16 pairs to FP32 and use scalar FMA
                    x0 = _bf16_to_f32(x_word, 0)
                    x1 = _bf16_to_f32(x_word, 1)
                    g0 = _bf16_to_f32(g_word, 0)
                    g1 = _bf16_to_f32(g_word, 1)
                    u0 = _bf16_to_f32(u_word, 0)
                    u1 = _bf16_to_f32(u_word, 1)
                    g_cur = arith.addf(
                        g_cur, arith.addf(arith.mulf(x0, g0), arith.mulf(x1, g1))
                    )
                    u_cur = arith.addf(
                        u_cur, arith.addf(arith.mulf(x0, u0), arith.mulf(x1, u1))
                    )

            scf.YieldOp([g_cur, u_cur])

        gate_sum = _butterfly_reduce(for_op.results[0])
        up_sum = _butterfly_reduce(for_op.results[1])

        # -- Epilogue (lane 0 only) --------------------------------------------
        lane_zero = arith.cmpi(CmpIPredicate.eq, lane_i32, arith.constant(0, type=i32))
        if_op = scf.IfOp(lane_zero)
        with ir.InsertionPoint(if_op.then_block):
            # silu(x) = x / (1 + exp(-x)), fast: exp2(-log2(e) * x)
            neg_log2e = arith.constant(-1.4426950408889634, type=f32)
            t = arith.mulf(gate_sum, neg_log2e)
            emu = rocdl.exp2(f32, t)
            c1f = arith.constant(1.0, type=f32)
            den = arith.addf(c1f, emu)
            sig = rocdl.rcp(f32, den)
            silu_g = arith.mulf(gate_sum, sig)
            out_f32 = arith.mulf(silu_g, up_sum)
            out_bf16 = arith.truncf(bf16, out_f32)

            out_elem = (token_b * TOPK_i32 + expert_k) * INTER_i32 + neuron_j
            buffer_ops.buffer_store(out_bf16, out_rsrc, out_elem)
            scf.YieldOp([])

    # -- @flyc.jit launch wrapper ----------------------------------------------
    _k = _kernel

    @flyc.jit
    def launch(
        arg_inter_out: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w_gate: fx.Pointer,
        arg_w_up: fx.Pointer,
        arg_router_ids: fx.Pointer,
        i32_B: fx.Int32,
        i32_TOPK: fx.Int32,
        i32_INTER: fx.Int32,
        i32_HIDDEN: fx.Int32,
        i32_E: fx.Int32,
        stream: fx.Stream,
    ):
        idx_t = ir.IndexType.get()
        B_idx = arith.index_cast(idx_t, i32_B.ir_value())
        TOPK_idx = arith.index_cast(idx_t, i32_TOPK.ir_value())
        INTER_idx = arith.index_cast(idx_t, i32_INTER.ir_value())
        grid_x = B_idx * TOPK_idx * INTER_idx
        _k(
            arg_inter_out,
            arg_x,
            arg_w_gate,
            arg_w_up,
            arg_router_ids,
            i32_B,
            i32_TOPK,
            i32_INTER,
            i32_HIDDEN,
            i32_E,
        ).launch(grid=(grid_x, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)

    return launch
