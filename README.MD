## Lộ trình học Machine Learning, Deep Learning cho người mới bắt đầu

Tôi đã từng học Machine Learning trong vòng 2 tháng và tôi tin bạn cũng có thể làm được. 

Lộ trình sẽ giúp bạn nắm chắc công nghệ này từ cơ bản đến nâng cao, xây dựng Machine Learning model từ python thuần cho đến các thư viện cao cấp như TensorFlow hay Keras. Đi sâu phân tích bản chất vấn đề là giá trị cốt lõi của khóa học này.

**P/S:** Hãy để lại **1 star** để team có động lực xuất bản các phần tiếp theo và cũng đừng quên chia sẻ tới bạn bè của bạn.

Mình sẽ là một trong các lecturer của lớp VietAI offline tại Hanoi diễn ra giữa tháng 11. Nếu bạn quan tâm có thể đăng ký [qua đây](https://docs.google.com/forms/d/e/1FAIpQLSf6QOy6FhxGmot4mkgt7A0Qk_II7V9Y1Z0zxS8SMmtwYR1sgQ/viewform). 


### 0. Mục lục
----
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/tree/master/math">1. Kiến thức toán học cần thiết</a> (Hoàn tất)
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#2-kỹ-năng-lập-trình-python">2. Kỹ năng lập trình Python</a> (Hoàn tất)
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#3-thư-viện-numpy-và-tensorflow">3. Thư viện Numpy và TensorFlow</a> (Hoàn tất)
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#4-bài-toán-hồi-quy-regression">4. Bài toán hồi quy (Regression)</a> (Hoàn tất)
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#5-bài-toán-phân-loại-classification">5. Bài toán phân loại (Classification)</a> (Chưa Hoàn tất)
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#6-xây-dựng-mô-hình-decision-trees-và-random-forests">6. Xây dựng mô hình Decision Trees và Random Forests </a> (Chưa Hoàn tất)
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#7-xây-dựng-mạng-neural-network">7. Xây dựng mạng Neural Network</a> (Đang tiến hành)
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#8-xây-dựng-mạng-convolutional-neural-network-cnn">8. Xây dựng mạng Convolutional Neural Network (CNN)</a> (Chưa Hoàn tất)
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#9-xây-dựng-mạng-recurrent-neural-network-rnn">9. Xây dựng mạng Recurrent Neural Network (RNN)</a> (Chưa Hoàn tất)
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#10-triển-khai-deploy-machine-learning-model-trên-production">10. Triển khai (Deploy) Machine Learning Model trên Production</a> (Chưa Hoàn tất)
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#11-machine-learning-trên-trình-duyệt-và-tensorflowjs">11. Machine Learning trên trình duyệt và TensorFlowJS</a> (Chưa Hoàn tất)
        
### 3. Thư viện Numpy và TensorFlow
----

   - **a. Numpy**
   
        - Giới thiệu về NumPy
            
            - Trang chủ của [NumPy](http://www.numpy.org/)
            - NumPy là viết tắt của **"Numerical Python"**, là một thư viện chuyên để xử lý, tính toán vector, ma trận.
            - NumPy thường được sử dụng cùng [SciPy](https://www.scipy.org/) và [Matplotlib](https://matplotlib.org/) để thay thế cho [MatLab](https://www.mathworks.com/products/matlab.html) vô cùng đắt đỏ.
            - Được viết bằng Python và C nên tốc độ thực thi tốt.
        - Các APIs
            - [Mảng NumPy - NumPy Array](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/array.ipynb)
            - [Truy cập thông qua chỉ mục và cắt Array - Array Indexing và Array Slicing](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/indexing%26slicing.ipynb)
            - [Phép lan truyền - Broadcasting](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/broadcasting.ipynb)
            - [Phép lặp - Iterating](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/iterating.ipynb)
            - [Các thao tác trên Array - Array Manipulation](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/manipulation.ipynb)
            - [Đại số tuyến tính - Linear Algebra](https://github.com/bangoc123/learn-machine-learning-in-two-months/tree/master/numpy/linear-algebra)
                - [Phép nhân vô hướng](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/linear-algebra/products.ipynb)
                - [Norm ma trận](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/linear-algebra/norms.ipynb)
                - [Phép cộng ma trận](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/linear-algebra/addition.ipynb)
            
   - **b. TensorFlow**
   
        Toàn bộ kiến thức cần biết về TensorFlow nằm [tại đây](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/tensorflow/README.MD)
              
### 4. Bài toán hồi quy (Regression)
----
  
  Toàn bộ mô hình này nằm [ở đây](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/models/linear-regression/README.MD)
            
### 5. Bài toán phân loại (Classification)
----

   - **a. Mô hình Hồi quy Logistic (Logistic Regression Model)**
        
        - Định nghĩa
        - Mô phỏng thực tế
        
   - **b. Phân loại 2 lớp (Binary Classification)**
        
        - Định nghĩa
        - Hàm phi tuyến sigmoid
        - Hàm mất mát (Loss Function)
        - Thuật toán Gradient Descent
        
   - **c. Phân loại nhiều lớp (Multiclass Classification)**
        
        - Định nghĩa
        - Hàm mất mát (Loss Function)
        - Thuật toán Gradient Descent
        
   - **d. Hiện tượng Overfitting và một số cách khắc phục**
   
   - **e. Đánh giá mô hình**

### 6. Xây dựng mô hình Decision Trees và Random Forests
----

   - **a. Mô hình Cây quyết định (Decision Trees)**
   
        - Định nghĩa
        - Mô phỏng thực tế
   
   - **b. Các thuật toán**
        
        - ID3
        - C4.5
        - C5.0
        - CART
        
   - **c. Triển khai thuật toán CART**
    
        - Câu hỏi (Question)
        
   - **d. Bài toán phân loại ô tô**
   
        - Vấn đề
        - Giải quyết bằng mô hình Cây quyết định
        
   - **e. Mô hình Random Forests**
   
        - Định nghĩa
        
### 7. Xây dựng mạng Neural Network
   Tất cả nội dung về Neural Network [tại đây](https://github.com/bangoc123/learn-machine-learning-in-two-months/tree/master/models/nn).

### 8. Xây dựng mạng Convolutional Neural Network (CNN)
   Toàn bộ kiến thức của CNN nằm [tại đây](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/computer-vision/CNN.MD) 
    
### 9. Xây dựng mạng Recurrent Neural Network (RNN)
----

   **a. Language Models**
    
   **b. Recurrent Neural Network (RNN)**
   
   - Giới thiệu RNN
   - Cấu trúc mạng RNN
   - Gate Recurrent Unit (GRU)
   - Long Short-Term Memory (LSTM)
   
   **b. Vấn đề thường gặp và cách giải quyết**
   - Vanishing
   - Expoding Gradient Descent
        
### 10. Triển khai (Deploy) Machine Learning Model trên Production
----

   **a. Triển khai Tensorflow phân tán (Distributed TensorFlow)**
   
   **b. TensorFlow với AWS S3**
   
   **c. TensorFlow Serving**
   - Giới thiệu
   - Kiến trúc
   - Cài đặt
   - Restful API
   - Xây dựng TensorFlow ModelServer
   - Triển khai Inception Model với TensorFlow và Kubernetes
        

### 11. Machine Learning trên trình duyệt và TensorFlowJS
----

Phần này được mô phỏng chủ yếu trên bài nói của tôi tại Google I/O Extended Hanoi 2018. Slide và code của bài nói này nằm [tại đây](https://github.com/bangoc123/tensorflow-js). 

   - **a. Lịch sử TensorFlowJS**
   - **b. Các API chính của TensorFlowJS**
   - **c. Xây dựng mô hình Hồi quy tuyến tính**
   - **d. Xây dựng mạng Convolutional Neural Network(CNN)**
   - **e. MobileNet**
   - **f. Transfer Learning**

   
