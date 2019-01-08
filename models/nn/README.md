### 7. Xây dựng mạng Neural Network
----
 - **a. Neural Network**

      - Định nghĩa
      - Mô phỏng thực tế

 - **b. Hàm kích hoạt phi tuyến**

      - Sigmoid
      - Tanh
      - ReLU


 - **c. Thuật toán lan truyền ngược (backpropagation)**

      - Đạo hàm riêng (Partial Derivative)

          <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;f(x,&space;y,&space;z)&space;=&space;(x&plus;y)z" title="\large f(x, y, z) = (x+y)z" />

          <br>

          <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;q&space;=&space;x&space;&plus;&space;y&space;\rightarrow&space;f=qz" title="\large q = x + y \rightarrow f=qz" />
          <br>

          <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\frac{\partial&space;q}{\partial&space;x}&space;=&space;1,&space;\frac{\partial&space;q}{\partial&space;y}&space;=&space;1,&space;\frac{\partial&space;f}{\partial&space;q}&space;=&space;z&space;=&space;-4" title="\large \frac{\partial q}{\partial x} = 1, \frac{\partial q}{\partial y} = 1, \frac{\partial f}{\partial q} = z = -4" />

          <br>

          <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\frac{\partial&space;f}{\partial&space;x}&space;=&space;\frac{\partial&space;f}{\partial&space;q}\frac{\partial&space;q}{\partial&space;x}&space;=&space;-4" title="\large \frac{\partial f}{\partial x} = \frac{\partial f}{\partial q}\frac{\partial q}{\partial x} = -4" />

          <br>

          <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\frac{\partial&space;f}{\partial&space;y}&space;=&space;\frac{\partial&space;f}{\partial&space;q}\frac{\partial&space;q}{\partial&space;y}&space;=&space;-4" title="\large \frac{\partial f}{\partial y} = \frac{\partial f}{\partial q}\frac{\partial q}{\partial y} = -4" />

          <br>

          <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\frac{\partial&space;f}{\partial&space;z}&space;=&space;q&space;=&space;3" title="\large \frac{\partial f}{\partial z} = q = 3" />

          <br>


      - Ôn lại đạo hàm hàm hợp (Chain Rule)
      - Công thức
      - Mô phỏng thực tế

 - **d. Thuật toán tối ưu Loss Function (Optimization Algorithms)**
      - Thuật toán Gradient Descent

 - **e. Một số vấn đề khi huấn luyện Neural Network**
