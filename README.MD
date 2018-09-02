## Lộ trình học Machine Learning, Deep Learning cho người mới bắt đầu

Tôi đã từng học Machine Learning trong vòng 2 tháng và tôi tin bạn cũng có thể làm được. 

Lộ trình sẽ giúp bạn nắm chắc công nghệ này từ cơ bản đến nâng cao, xây dựng Machine Learning model từ python thuần cho đến các thư viện cao cấp như TensorFlow hay Keras. Đi sâu phân tích bản chất vấn đề là giá trị cốt lõi của khóa học này.

**P/S:** Hãy để lại **1 star** để team có động lực xuất bản các phần tiếp theo và cũng đừng quên chia sẻ tới bạn bè của bạn.


### 0. Mục lục
----
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#1-kiến-thức-toán-học-cần-thiết">1. Kiến thức toán học cần thiết</a>
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#2-kỹ-năng-lập-trình-python">2. Kỹ năng lập trình Python</a>
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#3-thư-viện-numpy-và-tensorflow">3. Thư viện Numpy và TensorFlow</a>
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#4-bài-toán-hồi-quy-regression">4. Bài toán hồi quy (Regression)</a>
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#5-bài-toán-phân-loại-classification">5. Bài toán phân loại (Classification)</a>
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#6-xây-dựng-mô-hình-decision-trees-và-random-forests">6. Xây dựng mô hình Decision Trees và Random Forests </a>
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#7-xây-dựng-mạng-neural-network">7. Xây dựng mạng Neural Network</a>
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#8-xây-dựng-mạng-convolutional-neural-network-cnn">8. Xây dựng mạng Convolutional Neural Network (CNN)</a>
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#9-xây-dựng-mạng-recurrent-neural-network-rnn">9. Xây dựng mạng Recurrent Neural Network (RNN)</a>
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#10-triển-khai-deploy-machine-learning-model-trên-production">10. Triển khai (Deploy) Machine Learning Model trên Production</a>
- <a href="https://github.com/bangoc123/learn-machine-learning-in-two-months/#11-machine-learning-trên-trình-duyệt-và-tensorflowjs">11. Machine Learning trên trình duyệt và TensorFlowJS</a>
    
### 1. Kiến thức toán học cần thiết 
----
- **a. Đại số tuyến tính**

    Nếu các bạn có nhiều thời gian và sự kiên trì thì có thể học toàn bộ khóa này [MIT 18.06 Linear Algebra, Spring 2005
](https://www.youtube.com/watch?v=ZK3O402wf1c&list=PLE7DDD91010BC51F8). Nhưng đối với bản thân tôi có những phần trong khóa này đến bây giờ tôi vẫn chưa dùng đến. Vì thế nếu **bạn không có nhiều thời gian, muốn tăng nhanh tốc độ** thì có thể học theo từng phần tôi nhấn mạnh ở phía dưới đây.
    - **Scalar/Vector**
        - Giới thiệu vector/scalar và các thành phần của chúng: [Vector basics - Khan Academy](https://www.khanacademy.org/math/precalculus/vectors-precalc#vector-basic) 
    - **Ma trận (Matrix)**
        - Giới thiệu về ma trận: [Introduction to matrices - Khan Academy](https://www.khanacademy.org/math/precalculus/precalc-matrices#intro-to-matrices)
    - **Chuyển vị ma trận**
        - Cách chuyển vị ma trận và những vấn đề liên quan: [Transpose of a matrix - Khan Academy](https://www.khanacademy.org/math/linear-algebra/matrix-transformations#matrix-transpose) 
    - **Norm Vector**
        - Norm L1/L2: [Vector Norms](https://youtu.be/5fN2J8wYnfw)
    - **Tensor**
        - Giới thiệu về Tensor: [Tensors for Beginners 0: Tensor Definition](https://youtu.be/TvxmkZmBa-k)
    - **Các phép toán với ma trận**
        - Phép cộng ma trận
            -  Phương pháp cộng/trừ ma trận: [Matrix addition and subtraction | Matrices | Precalculus | Khan Academy
](https://youtu.be/WR9qCSXJlyY)
        - Phép nhân ma trận
            - Các phương pháp nhân ma trận: [Lec 3 | MIT 18.06 Linear Algebra, Spring 2005](https://youtu.be/FX4C-JpTFgY) 
        - Tích Hadamard/Element-Wise
            -  Phương pháp tính tích Element-Wise: [Element-Wise Multiplication and Division of Matrices
](https://youtu.be/2GPZlRVhQWY)
    - **Ma trận đơn vị**
        - Miêu tả ma trận đơn vị: [Identity matrix | Matrices | Precalculus | Khan Academy
](https://youtu.be/3cnIa0fYJkY)
    - **Ma trận nghịch đảo**
        -  Phương pháp tính ma trận nghịch đảo: [Lec 3 | MIT 18.06 Linear Algebra, Spring 2005](https://youtu.be/FX4C-JpTFgY?t=21m14s)
- **b. Đạo hàm**

    - Đây là series kinh điển để nhắc lại kiến thức đạo hàm của bạn. [Essence of calculus - 3Blue1Brown](https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

    - Anh Tiệp có một bài rất đầy đủ về đạo hàm ở đây: [Machine Learning cơ bản - Phần Toán](https://machinelearningcoban.com/math/). Hãy thực hành tính toán phần 3.5, tôi đảm bảo bạn sẽ nắm chắc được đạo hàm trên vector và ma trận.
    - Ngoài phần cơ bản quan trọng thì việc nắm thuần thục **Chain Rule** và **Production Rule** là rất quan trọng đặc biệt là dành cho thuật toán **Backpropagation** trong **Deep Learing**. Bạn hãy xem kỹ video này: [Visualizing the chain rule and product rule | Essence of calculus, chapter 4
](https://youtu.be/YG15m2VwSjA?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

- **c. Lý thuyết xác suất**
    - **Các khái niệm cơ bản**
        - Những khái niệm cơ bản: [Basic theoretical probability](https://www.khanacademy.org/math/statistics-probability/probability-library#basic-theoretical-probability) 
        - Xác suất sử dụng không gian mẫu: [Probability using sample spaces](https://www.khanacademy.org/math/statistics-probability/probability-library#probability-sample-spaces)
        - Tiên đề xác suất: [Axioms of Probability](https://youtu.be/xuv6BCR-iNc)
        - Các loại xác suất
            - Xác suất có điều kiện: 
                - Giới thiệu: [Dependent probability introduction | Probability and Statistics | Khan Academy
](https://youtu.be/VjLEoo3hIoM)
                - Các ví dụ liên quan: [Dependent probability example | Probability and Statistics | Khan Academy
](https://youtu.be/xPUm5SUVzTE)
                - Công thức Bayes: [CRITICAL THINKING - Fundamentals: Bayes' Theorem](https://youtu.be/OqmJhPQYRc8)
            - Xác suất độc lập: [Compound probability of independent events | Probability and Statistics | Khan Academy
](https://youtu.be/xSc4oLA9e8o)
        - Biến ngẫu nhiên và phân phối xác suất: [Biến ngẫu nhiên và phân phối xác suất](https://dominhhai.github.io/vi/2017/10/prob-rand-var/#2-1-h%C3%A0m-kh%E1%BB%91i-x%C3%A1c-su%E1%BA%A5t-c%E1%BB%A7a-bi%E1%BA%BFn-r%E1%BB%9Di-r%E1%BA%A1c)
        
### 2. Kỹ năng lập trình Python
----
Python là ngôn ngữ được dùng nhiều nhất để làm Machine Learning vì tính đơn giản gọn nhẹ của nó. Nhưng để đưa vào Production thì tôi nghĩ Javascript cũng là một lựa chọn không tồi. Tôi sẽ chia sẻ về Machine Learning với Javascript trong các phần tiếp theo. 

- **a. Cài đặt Python và các thư viện cần thiết**:   
    - [Windows](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/essential-installs/vn-windows-python-and-libraries-installations.MD)  
        - Python
        - Pip
        - Jupyter Notebook
    - [Mac](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/essential-installs/vn-mac-python-and-libraries-installations.MD)
        - Python
        - Pip
        - Jupyter Notebook
- **b. Tính chất đặc điểm**

    Python là ngôn ngữ thông dịch có:
    - Điểm mạnh: 
        - Dễ viết/ Dễ đọc
        - Quy trình phát triển phần mềm nhanh vì dòng lệnh được thông dịch thành mã máy và thực thi ngay lập tức
        - Có nhiều thư viện mạnh để tính toán cũng như làm Machine Learning như Numpy, Sympy, Scipy, Matplotlib, Pandas, TensorFlow, Keras, vv.
    - Điểm yếu:
        - Mang đầy đủ điểm yếu của các ngôn ngữ thông dịch như tốc độ chậm, tiềm tàng lỗi trong quá trình thông dịch, source code dễ dàng bị dịch ngược.
        - Ngôn ngữ có tính linh hoạt cao nên thiếu tính chặt chẽ.
- **c. Các hàm dựng sẵn và kiểu dữ liệu trên Python**
    - [Các hàm dựng sẵn trong Python](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/python-tutorials/built-in-function.ipynb)
    - Các kiểu dữ liệu trong Python
    
        Phần này được thiết kế dựa trên khóa [Data Structures & Algorithms in Python - Udacity](https://classroom.udacity.com/courses/ud513) nhưng tôi sẽ đi sâu hơn vào các ví dụ thực tế hơn là chỉ dừng lại ở lý thuyết. Bên cạnh đó tôi sẽ giúp các bạn nắm rõ một số Data Structure quan trọng trong Machine Learning như **Graph** và **Tree**.
        - [Integer, Float và Boolean](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/python-tutorials/integer-float-boolean.ipynb)
        - [String](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/python-tutorials/string.ipynb)
        - List-Based Collection
            - [List](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/python-tutorials/list.ipynb)
            - [Linked List](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/python-tutorials/linked-list.ipynb)
            - [Stack](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/python-tutorials/stack.ipynb)
            - [Queue](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/python-tutorials/queue.ipynb)
            - [Tuple](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/python-tutorials/tuple.ipynb)
        - [Dictionary](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/python-tutorials/dictionary.ipynb)
        - [Set](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/python-tutorials/set.ipynb)
        - [Graph](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/python-tutorials/graph.ipynb)
- **d. Vòng lặp**
    - [Hàm While và For In](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/python-tutorials/loop.ipynb)
- **e. Hàm**
    - [Hàm và các vấn đề liên quan](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/python-tutorials/function.ipynb)
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
                - [Phép nhân ma trận](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/linear-algebra/products.ipynb)
                - [Norm ma trận](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/linear-algebra/norms.ipynb)
            
   - **b. TensorFlow**

### 4. Bài toán hồi quy (Regression)
----

   - **a. Mô hình Hồi quy tuyến tính (Linear Regression Model)**
        - [Bài toán đơn giản](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/models/linear-regression/README.MD/#1-b%C3%A0i-to%C3%A1n-%C4%91%C6%A1n-gi%E1%BA%A3n)
        - Định nghĩa
        - Mô phỏng thực tế
        
            ![](https://lh6.googleusercontent.com/xQMPlPjv845-gDphAAcyNLJufG_774prU974Zn00WP8Hsfandysppi3OiizibH_fdiYhODkhrmTKOc_HtoC74HHb6s6L0Bj4n1jv_P-xJWJTcSDgsO5ncISfO_SKLGq-8W3o2lVgf30 )
        
   - **b. Hàm mất mát (Loss Function)**
   
        - Định nghĩa
        - Mô phỏng thực tế
        
   - **c. Thuật toán tối ưu Loss Function (Optimization Algorithms)**
   
        - Normal Equation
            - Công thức toán học
            - Điểm mạnh
            - Điểm yếu
        - Gradient Descent
            - Công thức toán học
            - Điểm mạnh
            - Điểm yếu
            
   - **d. Bài toán thực tế**
   
        - Bài toán đo lường Hiệu quả sản xuất
            - Vấn đề
            - Giải quyết bằng mô hình Hồi quy tuyến tính
            
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
----

   - **a. Neural Network**
   
        - Định nghĩa
        - Mô phỏng thực tế
   
   - **b. Hàm kích hoạt phi tuyến**
        
        - Sigmoid
        - Tanh
        - ReLU
   
   
   - **c. Thuật toán lan truyền ngược (backpropagation)**
    
        - Ôn lại đạo hàm hàm hợp (Chain Rule)
        - Công thức
        - Mô phỏng thực tế
        
   - **d. Thuật toán tối ưu Loss Function (Optimization Algorithms)**
        - Thuật toán Gradient Descent
        
   - **e. Một số vấn đề khi huấn luyện Neural Network**

### 8. Xây dựng mạng Convolutional Neural Network (CNN)
----

### 9. Xây dựng mạng Recurrent Neural Network (RNN)
----

### 10. Triển khai (Deploy) Machine Learning Model trên Production
----

### 11. Machine Learning trên trình duyệt và TensorFlowJS
----

Phần này được mô phỏng chủ yếu trên bài nói của tôi tại Google I/O Extended Hanoi 2018. Slide và code của bài nói này nằm [tại đây](https://github.com/bangoc123/tensorflow-js). 

   - **a. Lịch sử TensorFlowJS**
   - **b. Các API chính của TensorFlowJS**
   - **c. Xây dựng mô hình Hồi quy tuyến tính**
   - **d. Xây dựng mạng Convolutional Neural Network(CNN)**
   - **e. MobileNet**
   - **f. Transfer Learning**

   
