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

    Matrix *scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(current_query, k_stack, scores);

    Matrix *softmax_mat = nullptr;
    for (size_t row = 0; row <= i; ++row) {
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

      if (softmax_mat == nullptr) {
        softmax_mat = matrix_memory_allocator.Allocate("softmax_init");
        gpu_sim.Copy(row_soft, softmax_mat, kInSharedMemory);
        gpu_sim.ReleaseMatrix(row_soft);
      } else {
        Matrix *softmax_next = matrix_memory_allocator.Allocate("softmax_next");
        gpu_sim.Concat(softmax_mat, row_soft, softmax_next, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_mat);
        gpu_sim.ReleaseMatrix(row_soft);
        softmax_mat = softmax_next;
      }
    }

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

    Matrix *answer = matrix_memory_allocator.Allocate("answer");
    gpu_sim.MatMul(softmax_mat, v_stack, answer);

    gpu_sim.ReleaseMatrix(scores);
    gpu_sim.ReleaseMatrix(k_stack);
    gpu_sim.ReleaseMatrix(softmax_mat);
    gpu_sim.ReleaseMatrix(v_stack);

    gpu_sim.MoveMatrixToGpuHbm(answer);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
