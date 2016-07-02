__kernel void transMatrixMult(__global const float *left, __global const float *right, __global float *target, const int leftHeight, const int leftWidth, const int rightHeight) {   int row = get_global_id(0);
  int col = get_global_id(1);
  __global float* a = left + row * leftWidth;  __global float* b = right + leftWidth * col;  float sum = 0;  int n;  for(n = 0; n < leftWidth; n++) {    sum += left[row * leftWidth + n] * right[leftWidth * col + n];  }  target[row * rightHeight + col] = sum;
}