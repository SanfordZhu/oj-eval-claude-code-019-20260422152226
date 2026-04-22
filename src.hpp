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

    // Build scores (i+1 x i+1) column by column without stacking K
    Matrix *scores = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      Matrix *k_copy = matrix_memory_allocator.Allocate("k_copy");
      gpu_sim.Copy(keys[j], k_copy, kInSharedMemory);
      gpu_sim.Transpose(k_copy, kInSharedMemory); // (d,1)
      Matrix *col = matrix_memory_allocator.Allocate("score_col");
      gpu_sim.MatMul(current_query, k_copy, col); // (i+1,1)
      gpu_sim.ReleaseMatrix(k_copy);

      if (scores == nullptr) {
        scores = matrix_memory_allocator.Allocate("scores_init");
        gpu_sim.Copy(col, scores, kInSharedMemory);
        gpu_sim.ReleaseMatrix(col);
      } else {
        Matrix *new_scores = matrix_memory_allocator.Allocate("scores_next");
        gpu_sim.Concat(scores, col, new_scores, 1, kInSharedMemory);
        gpu_sim.ReleaseMatrix(scores);
        gpu_sim.ReleaseMatrix(col);
        scores = new_scores;
      }
    }

    // Softmax rows
    Matrix *softmax_mat = nullptr;
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

    // Build V_stack (i+1 x d) once
    Matrix *v_stack = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(values[j]);
      if (v_stack == nullptr) {
        v_stack = matrix_memory_allocator.Allocate("v_init");
        gpu_sim.Copy(values[j], v_stack, kInSharedMemory);
      } else {
        Matrix *v_next = matrix_memory_allocator.Allocate("v_next");
        gpu_sim.Concat(v_stack, values[j], v_next, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(v_stack);
        v_stack = v_next;
      }
    }

    // Answer = softmax_mat * v_stack
    Matrix *answer = matrix_memory_allocator.Allocate("answer");
    gpu_sim.MatMul(softmax_mat, v_stack, answer);

    // Cleanup
    gpu_sim.ReleaseMatrix(scores);
    gpu_sim.ReleaseMatrix(softmax_mat);
    gpu_sim.ReleaseMatrix(v_stack);

    // Commit
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
