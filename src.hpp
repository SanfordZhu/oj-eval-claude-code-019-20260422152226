#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Build K_stack (i+1 x d) in SRAM
    Matrix *k_stack = matrix_memory_allocator.Allocate("k_stack_init");
    gpu_sim.MoveMatrixToSharedMem(keys[0]);
    gpu_sim.Copy(keys[0], k_stack, kInSharedMemory);
    for (size_t j = 1; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      Matrix *k_next = matrix_memory_allocator.Allocate("k_stack_next");
      gpu_sim.Concat(k_stack, keys[j], k_next, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(k_stack);
      k_stack = k_next;
    }

    gpu_sim.Transpose(k_stack, kInSharedMemory);

    // scores = Q * K^T in SRAM
    Matrix *scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(current_query, k_stack, scores);

    // Build V_stack (i+1 x d) in SRAM once
    Matrix *v_stack = matrix_memory_allocator.Allocate("v_stack_init");
    gpu_sim.MoveMatrixToSharedMem(values[0]);
    gpu_sim.Copy(values[0], v_stack, kInSharedMemory);
    for (size_t j = 1; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(values[j]);
      Matrix *v_next = matrix_memory_allocator.Allocate("v_stack_next");
      gpu_sim.Concat(v_stack, values[j], v_next, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(v_stack);
      v_stack = v_next;
    }

    // Build answer row-by-row: row_soft (1 x i+1) * V_stack (i+1 x d)
    Matrix *answer_stack = nullptr;
    for (size_t row = 0; row <= i; ++row) {
      Matrix *row_vec = matrix_memory_allocator.Allocate("row_vec");
      gpu_sim.GetRow(scores, row, row_vec, kInSharedMemory);
      Matrix *row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.MatExp(row_vec, row_exp);
      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_exp, row_sum);
      Matrix *row_soft = matrix_memory_allocator.Allocate("row_soft");
      gpu_sim.MatDiv(row_exp, row_sum, row_soft);
      gpu_sim.ReleaseMatrix(row_vec);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);

      Matrix *row_ans = matrix_memory_allocator.Allocate("row_ans");
      gpu_sim.MatMul(row_soft, v_stack, row_ans);
      gpu_sim.ReleaseMatrix(row_soft);

      if (answer_stack == nullptr) {
        answer_stack = matrix_memory_allocator.Allocate("answer_init");
        gpu_sim.Copy(row_ans, answer_stack, kInSharedMemory);
        gpu_sim.ReleaseMatrix(row_ans);
      } else {
        Matrix *ans_next = matrix_memory_allocator.Allocate("answer_next");
        gpu_sim.Concat(answer_stack, row_ans, ans_next, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(answer_stack);
        gpu_sim.ReleaseMatrix(row_ans);
        answer_stack = ans_next;
      }
    }

    // cleanup intermediates
    gpu_sim.ReleaseMatrix(scores);
    gpu_sim.ReleaseMatrix(k_stack);
    gpu_sim.ReleaseMatrix(v_stack);

    // Move answer to HBM and commit
    gpu_sim.MoveMatrixToGpuHbm(answer_stack);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer_stack);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
