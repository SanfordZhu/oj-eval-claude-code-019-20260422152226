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

    // Build answer row-by-row to reduce SRAM and avoid large MatMul
    Matrix *answer_stack = nullptr;
    for (size_t row = 0; row <= i; ++row) {
      // softmax on this row
      Matrix *row_mat = matrix_memory_allocator.Allocate("row");
      gpu_sim.GetRow(scores, row, row_mat, kInSharedMemory);
      Matrix *row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.MatExp(row_mat, row_exp);
      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_exp, row_sum);
      Matrix *row_soft = matrix_memory_allocator.Allocate("row_soft");
      gpu_sim.MatDiv(row_exp, row_sum, row_soft);
      gpu_sim.ReleaseMatrix(row_mat);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);

      // weighted sum over V[j]
      Matrix *acc = nullptr;
      for (size_t j = 0; j <= i; ++j) {
        gpu_sim.MoveMatrixToSharedMem(values[j]);
        Matrix *weight = matrix_memory_allocator.Allocate("weight");
        gpu_sim.GetColumn(row_soft, j, weight, kInSharedMemory); // 1x1
        Matrix *weighted = matrix_memory_allocator.Allocate("weighted_v");
        gpu_sim.MatMulNum(values[j], weight, weighted);
        gpu_sim.ReleaseMatrix(weight);
        if (acc == nullptr) {
          acc = matrix_memory_allocator.Allocate("acc_init");
          gpu_sim.Copy(weighted, acc, kInSharedMemory);
          gpu_sim.ReleaseMatrix(weighted);
        } else {
          Matrix *new_acc = matrix_memory_allocator.Allocate("acc_next");
          gpu_sim.MatAdd(acc, weighted, new_acc);
          gpu_sim.ReleaseMatrix(acc);
          gpu_sim.ReleaseMatrix(weighted);
          acc = new_acc;
        }
      }

      // append acc as one row to answer_stack
      if (answer_stack == nullptr) {
        answer_stack = matrix_memory_allocator.Allocate("answer_init");
        gpu_sim.Copy(acc, answer_stack, kInSharedMemory);
        gpu_sim.ReleaseMatrix(acc);
      } else {
        Matrix *ans_next = matrix_memory_allocator.Allocate("answer_next");
        gpu_sim.Concat(answer_stack, acc, ans_next, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(answer_stack);
        gpu_sim.ReleaseMatrix(acc);
        answer_stack = ans_next;
      }

      gpu_sim.ReleaseMatrix(row_soft);
    }

    // cleanup intermediates
    gpu_sim.ReleaseMatrix(scores);
    gpu_sim.ReleaseMatrix(k_stack);

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
