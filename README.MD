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
        - [Thực hành với Numpy](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/array.ipynb)
    - **Ma trận (Matrix)**
        - Giới thiệu về ma trận: [Introduction to matrices - Khan Academy](https://www.khanacademy.org/math/precalculus/precalc-matrices#intro-to-matrices)
        - [Thực hành với Numpy](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/array.ipynb)
    - **Chuyển vị ma trận**
        - Cách chuyển vị ma trận và những vấn đề liên quan: [Transpose of a matrix - Khan Academy](https://www.khanacademy.org/math/linear-algebra/matrix-transformations#matrix-transpose) 
        - [Thực hành với Numpy](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/manipulation.ipynb)
    - **Norm Vector**
        - Norm L1/L2: [Vector Norms](https://youtu.be/5fN2J8wYnfw)
        - [Thực hành với Numpy](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/linear-algebra/norms.ipynb)
    - **Tensor**
        - Giới thiệu về Tensor: [Tensors for Beginners 0: Tensor Definition](https://youtu.be/TvxmkZmBa-k)
    - **Các phép toán với ma trận**
        - Phép cộng ma trận
            -  Phương pháp cộng/trừ ma trận: [Matrix addition and subtraction | Matrices | Precalculus | Khan Academy
](https://youtu.be/WR9qCSXJlyY)
            - [Thực hành với Numpy](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/linear-algebra/addition.ipynb)
        - Phép nhân ma trận
            - Các phương pháp nhân ma trận: [Lec 3 | MIT 18.06 Linear Algebra, Spring 2005](https://youtu.be/FX4C-JpTFgY) 
            - [Thực hành với Numpy](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/linear-algebra/products.ipynb)
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
                - [Phép nhân vô hướng](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/linear-algebra/products.ipynb)
                - [Norm ma trận](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/linear-algebra/norms.ipynb)
                - [Phép cộng ma trận](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/numpy/linear-algebra/addition.ipynb)
            
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
   - **a. Convolutional Neural Network**
   
        - Định nghĩa
        
            Convolutional Neural Networks tương đồng với mạng Neural Networks thông thường, bao gồm nhiều neuron với mục đích học được các tham số (**weights** và **biases**). Mạng này có: 
            - Đầu vào: Những bức ảnh dưới dạng Pixel
            - Đầu ra: Các lớp cần phân loại
            
            Ví dụ tôi muốn làm bài toán phân loại chó mèo, thì đầu vào của tôi là rất nhiều ảnh chó mèo và lớp đầu ra có 2 neuron đại diện tương ứng cho chó và mèo.
        - Mô phỏng thực tế
        <img src="https://lh3.googleusercontent.com/cO8U8cDrxBjLVAEjdAsSk9LOc3JQ0rdOZQVQPFkMPFpnA5GhW-Z1gV6TRmyp1iUKygCS8Haa2d7-RLl1xT-WPtF7rou4xy38pB48cTZrSp3iytGdLVfsABSwANInzy3nno6bvCh1EQ4" width="500"/>
        
        - Ứng dụng thực tế
        
        
            |Bài toán | Ảnh minh họa |
            | ------ | ------ |
            |    Phân loại đối tượng trong ảnh (Image Classification)  |   <img src="https://lh3.googleusercontent.com/PGxisLf4UnaOaIU9MRMtTRgemCg-IS4gGnM7ZbzmeuIYWlyIkfXAozbhkvP6dqIc5dZLWA_Czfr1f5Cg6LWCftIdlKJYkgDdhlbJ9ukqt7q60t8a9sQUdl_F9jRWB6VlwPp6AEhkFYePAQipd3Wq23C-sw89XJkQoDuswZGXpBepnUAS3vgfvPbnlR1XsLfXm8CGI7pechOlaHpjwsYxivpISFFb-0xqCMxPyOsOFtfmsQ-IxNIEql_1j-Sk9SUmJS5zqGwzJiwJLqVRpwGaenFCHWbvdOP3GBa7dIFUIemlkztpQkiQI_9-ROdCKo1OV9-jST_ka0SQj10C_7bDgWLsvhJa8lNBApQQIiab9kKv2Olj0VwXng29Ox59PN8YbwQ0Jb7KuFRQJ7yqXseHm_z3y1_HWn2bHLTAUWpM-7Gx6EFki_xLlhQFPiQKQ7G4kBf9U4JrqbfVRDktLqQoItaow7rTEcmm-amK0Yfpwl5ycpkoiLztC50i5T5Xx6Y4N7IeDP3KD1bpWqnQBdC6YWYLofD83HWN3gnYyAbAGLL4H3cILjIQlqdju_fNUeW7jl7_IYj_b3LTOU69tph5gwATcISQ_8giK22UxbBC4Gluyg_93F6vR8ndeYghNsjxto0o4yoRYuZXxj6XR1Knlfp4EvkX4FeSZGBZN17S5h4UxSrxsM5zEn_rRA=w540-h377-no" width="200">   |
            |   Phát hiện đối tượng (Object Detection)   |   <img src="https://lh3.googleusercontent.com/n3RK8DtyfnwInW-E_waItuHB7TK2jXPhTQrTm6E063jJ5yRU5qDL7jxyAmO-moC4CQoRtmxVaSxhfwAUCbc2F9rHu31b3nG4fM_mESlGiZb_pi2d6gbLMPG959xTg-Za8qHOiiXmL4avsPRZKWTQRm8o6LwXkranfkrbXqT2L8YRIJFQw4e35nRO4lwyNKLjd4y64CpbXGcgqWxZRPjd2KUrvdolK86BFHNTWfNW3yFHEfw6uIEIPLmUNX8HB7-voOK2Ztc_po6gtXv9atWlXDIQU6ZIwYTrHmlybDrRSf18_jtm5wc4ulS22rWv12cYvCIUJwledmmYDp4Lmp0hok9Gjjxlj3co7MSgchkneMIideJ_0R-KqAvKpc-d_pWmnqy4p4bVKF70_GApRhPnVKF72iAhSZuthUso8YmVb_qddMyIc4mfAqCqdpp0lUyTQu7z28IvfaxKP4TPYM-3K3NiQgmvQsRt76dOXRDLVUbp57WoPOvhoI_INJjL6PFNboxvvGDu6VCh97GIpzhVNCDQ2BpsHRRdM-k-8NwKX6s29thDYJEXn0baA_9KDZm9VP7KVs8aH9qTih1etPGHK6nVJvEDkMNwkhvDZ6DlfiYzdrHO0v0tXtWXi2L9nFdckukQYZ4HRRHejDKqh46pjn-H0OtGfLKS_nud2ZJs6ty37bII56G-nA_OAw=w1185-h667-no" width="200">   |
            |     Phụ đề cho ảnh (Image Captioning)   |   <img src="https://lh3.googleusercontent.com/0SJvkIuLb428Zqw25aoUUvOn2tpcNeLYXER73GRuXL6hIbkbhmPm7MGqduVMd-F8zntBIR5bZnOaiev78_JUUH4i9Kdu6wBR4P_auw0RPvqbw4voLdBBX3V8Nm52o3LQbmIjhwHrqNNu00AK4XMoXpVFw3M-gt4HWE5onKzmQ6iIl9hl85os29vgpRC4mkULB0gGWJErJM-DFvyRBbHCL9ew1EFYrzySv_vgk1cTX0z4SzeGSFVt5SZXtXwVYqqmsmJ2PWN5QBNH8AlL1wjHgf0tvGNCbyz6SsGi_k_cVa3TLMVr_axOLRutWWZwz79EErf9VayCgCCV3TYASaufJyELAPWXTqJzYrtUruagBeqJqNEAucYv99p6XOl2epWKqFX9El-_cWJKvIdFWMRZYTkrTnWQGjnWVns8ScWqBGhvtbc6Z82O22mZVWtx612iSXjpGaxa9KqiXHwPBfvWLBr4EM5N6X4KkZzxxMhIBSAkrrCNpxg_MQAbX90Ob6PA8tJZdY-J9wzhpgxed3Vc_1tMFTXRqq2cRz2n_HtmIHpBiNQoX3fPzpcf-ALQc-iw54PWWW960HD8eJ2aC1Eo_4eZGgP2JeCYo3PQskQ1t1J9oV4tnZ1h9d39a09izPIUun8kRrYxaHMl2QRGy5Pw055GqLdvjKRzxr9-xkH6rU9Ch9T8tALEGBWdkA=w429-h273-no" width="200">   |
            |    Phân đoạn ảnh (Image Segementation)   |   <img src="https://lh3.googleusercontent.com/t9xeGuHQ0QuYL_s9jBpiyHgOAH7BiHPvOJjKaP6aix1hDlvL1Mso_ZE--qu-1wcDNMFuBryqdUvWCvF4XdFh8q0_30lK_VxYrUTj900udz_Pgs3SZWLLULSfyNh4e0NTZXuNGvRgCk7UOeFQjn99XNHLxDt73ciLt3eIFjmXhI7JeVzl39xpSdmQytYlDWxhqB_iy3BlPJ6j-wTx47XbnVNOpXtlJZITNRW4tMp6Nd2W7VubebUzvz8R51_wv-nGPs3EBZoL9R0n1-beBJ-95kUOcc8pYEadH5w7NzP3dT0hku3jyUxnTtS7hD83ChEQeqqYywUGEYPxbeC681kyDGhNoeWz7jBaYgkBiqHnPsSkVh5VP2jD3KQtPqH9ZxSCKruOg-TiduBGS8l0doUWnsZbgE632L-W8-w157zFiksU-cZEMv3I89jXXlDcim9wvkOfHBgdxHGPYyPJvA6GWk7wRSUzP8RPIxi84diTIFlVFBOes3S_FUeUweBvY0-ILfhDKaqoDJxO_DMvZEEAtib-CFQNW1-3RCwaUUqwZXx0Vmj0YTvf79NuCrB6dgf4_Lqi6YCiB1FoouoGXD-IwqKDwC7RPiBax95-f-0ZtcTz3dewyl6PeqTADBijcps4F_UhjmMt6tU0jInS5eHnCDt_fGBIzLDtmmQZtBMUouoloulqV96LsORQbQ=w571-h357-no" width="200">   |
            |    Xử lý ảnh (Image Processing)  |   <img src="https://lh3.googleusercontent.com/0xRnycdVy4QnhRpQejtzjG8Ap9r2dpB5v-jCaBkZ0CDVMY6eZV7yktVtUMQl5HoZ54UqxIcu6cKc7276BgSgD1csIv-y1oFmFHZnAceVVYGh0eiq0XEIqbmCPd_d29pCDd3pnogf6bNUEXp_deQxkSogO7wFL9nYw70Oyrera1tgTFZ_mqAu4NUwuY1v6xyCQShuUE7p6g8bAaDlGbjnB3jPcfpq89UUEknLc7jBrv1HLq8lCEyZr45vfDY6J_CagUTxOciwuNisaPgeYyBFJ5BGUxKo5SVSplEvclRB9RcHUTM_55dinQsDOQlDItijkUrZaPAb4itLM8SBPJwreAJXezrxFEviPZnJSwxwMZLzPEXJi1ZcU4ccTVUtH5BugCjO2QfQM_NwrUkRNuzecz57AdZxXOF1esTYkgqJDlMFC3zoCIjZ0IXSEpczCOGhirNQ6ti87UNo88NeX8irAoiscfGH63xvQ_hOmiSdWnLbU2Nwd8LVmQWEknDu6wirG9JiOpzdZxqsBcsap-HOpCGpdwYF9Szqj5SClmKzmp8ll0WbV_kKHAwLYstFUPkp9QjE28ZettDhZh4PiB1LoEcs251EaNLvm8LFvAdw_bGrog4fy2I9uxSJcibbkNsBsDXnYEqklTTLlYSwuZNjvMeHBzj8JmzT896eJHrSlhNX3FAsdIfqBrnY-A=w365-h253-no" width="200">
     
        Ảnh từ [cs231](http://cs231n.github.io/classification/), [analyticsvidhya](https://www.analyticsvidhya.com/blog/2018/06/understanding-building-object-detection-model-python/), [iospress](https://content.iospress.com/articles/international-journal-of-hybrid-intelligent-systems/his246), [quora](https://www.quora.com/What-is-the-need-of-Image-segmentation), [slideshare](https://www.slideshare.net/sofianhw/digital-image-processing-70067047)
        - Quá trình mạng nhận diện một ảnh (Inference)
        
            - Bước 1: Ảnh sẽ được chuyển về dưới dạng ma trận các pixel
            - Bước 2: Ma trận của ảnh khi đi qua các lớp ẩn (hidden layer) sẽ được trích xuất đặc trưng cũng như biến đối phi tuyến thành ma trận kích thước nhỏ hơn nhưng chứa đựng nhiều đặc trưng của bức ảnh hơn.
            - Bước 3: Sau khi ma trận của ảnh được biến đổi, nó được đưa qua lớp cuối cùng là một layer 2 neuron, mỗi neuron này sẽ đại diện cho **xác suất** ảnh đưa vào là **chó** hoặc **mèo**.
      
   - **b. Tại sao lại cần dùng Convolutional Neural Network?**
        - Nhược điểm của mạng kết nối đầy đủ (Fully Connected Network)
        
            - Dễ bị Overfit do lượng tham số cần học quá lớn và data quá nhỏ
            - Mạng không phân loại được cùng một đối tượng nhưng ở vị trí khác nhau trong ảnh
            
            
            <img src="https://lh3.googleusercontent.com/cHir-OK_Gk435zNfvXGibIl-IWOSuZQgBseZRIJxsyxWwhzM776srBiE_ksI3nPPO-RCekJbpTTlYpUG3KVfIR681iEVUrBi1Am-7IqCiDWQ8e3nRWipw5Lzt9-piQnUn8l-OfnG0Ol_5AR_W94fe-ab_SuAhodkHzOG1k9Kc_VnRSwxT1Aj2kOs22PNmBsLHWpgbHLf-alzvbI-G1QKXWfFpSSNeymcEQnO0HcAp43CpDqnludLNLJrzEFsf01-kAb-Rmg6Rxk0GUl3lWEPJPHEf6Z9zz_v-7uhE_WBye0Si7zvLQRYZYpRilDToDJjnHm-cJwS3TckPpovrqESKd0XabuM1xDXqvjG5kVoVzujCujQfuPC4jkCHLlX9-hg5nDQu4GfopWS5zXfMspZKXRKDChbfDDHr_W06EULfFdWsXoaMrZczMukMEjnKX8bCuhIm53mAiADazfeIOjLYgmkn8lDEaOf5_PNqygWFIooxp1dhZxl4CoctRsWERI8r9PFhnj0AR8ZoUsj35PpEd2prk-nadf8ax5vkDh8o3soSiHIMKWiWZg34RJ-5z57tXFU_MyjTO7_YZ-mKjWbUoMPSGkO7qVuSF61wI48wvDJhjLgYuLK0G5125U7xPI9XXPyJ0cSDuAyz3ldNS5ehzDZyvzX-M_A5E1PS3t1AwmLMxI0pNRHopaYqQ=s800-no" width="200">
            
        - Ưu điểm của mạng CNN
            - Lượng tham số ít hơn nhiều so với mạng kết nối đầy đủ. Các lớp tích chập (CONV), các lớp Pooling giúp trích xuất đặc trưng của ảnh khiến ma trận ảnh đi qua mỗi lớp nhỏ dần đi, vì thế lượng tham số cần học cũng giảm theo.
            - Có tính kháng dịch chuyển. Vì loại đi những dữ liệu thừa có ảnh hưởng đến vị trí của đối tượng nên mạng nhận diện chính xác một đối tượng nằm ở các vị trí khác nhau trong một ảnh.
        
   - **c. Các mạng CNN thông dụng**
        - VGG 16
        - ResNet
        - GoogleNet
        
   - **d. Phép tích chập (Convolution)**
        - Công thức
        
            ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/271b9c48087f01b626627a423d8e9003aa765728)

           Mô phỏng công thức:

            <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/Convolution_of_box_signal_with_itself2.gif" width="300"/> 

            Hàm f(x) đại diện là ô vuông màu xanh và Hàm g(t-x) đại diện cho ô vuông màu đỏ. Tích chập (f*g)[n] thể hiện mức độ giao hoán của hai hàm f(x) và g(t-x) tại ví trí n.
            - Tại vị trí n mà 2 hàm giao hoán mạnh nhất, tức là 2 ô vuông trùng nhau thì giá trị tích chập là lớn nhất
            - Tại vị trí n mà 2 hàm không giao hoán, tức là 2 ô vuông không trùng nhau thì giá trị tích chập là bằng 0
            - Giá trị tích chập tăng dần tỉ lệ thuận với mức độ giao hoán của 2 hàm
            
            Sử dụng tích chập, ta có thể trích xuất tính chất từ ma trận của ảnh dựa trên những ma trận nhỏ(filter) cho sẵn. 
        
        - Tích chập 1D rời rạc
        - Tích chập 2D rời rạc
   - **e. Pooling**
        - Max-Pooling
        - Average-Pooling
   - **f. Cấu trúc mạng**
        
        <img src="http://cs231n.github.io/assets/cnn/convnet.jpeg" width="500">
        
        Ảnh từ [cs231 - Stanford](http://cs231n.github.io/convolutional-networks/#overview)
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

   
