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
        - TensorFlow
        - Annaconda
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
   
        TensorFlow là một thư viện mã nguồn mở dùng để training/inference machine learning model có thể sử dụng cho cả nghiên cứu và sản phẩm.
        TensorFlow được ghép bởi 2 từ **Tensor** và **Flow**. Tensor đại diện cho giá trị các biến, Flow là dòng chảy. TensorFlow mang hàm ý là dòng chảy các giá trị trong Graph. Và trái tim của TensorFlow chính là **Computational Graph**.
        
        - Computational Graphs
        
          Computational Graph tương tự một Graph thông thường gồm nhiều đỉnh và các cạnh. Nó dùng để biểu diễn các phép tính toán dựa trên ngôn ngữ của Graph.
        
            - Đỉnh (Node) đại diện cho biến đầu vào, phép tính toán

            - Các cạnh (Edge) đại điện cho dữ liệu truyền bên trong Graph
            
            ![](https://lh3.googleusercontent.com/3E6Uzj1965hMt0xVnWOsdgiZo3jwmQL4GG0pvQapBHjkVusakD-wYs5gNJTMUvPNR1W4kuTLGPSMFVjo_52x2oiDZGuYJ0IVbnq3_a0AJa_P8gNhk7_fP1nLn3oiZhR-D4JxoPbHlWngJ_ZPBTFrv_jxSYxGBVL5wYLE6ulnvkM-N_C0OxRJH4tXEKxgU2DwclhZuGFbvcvgDrwLylACYZI-B62YJLqaS-i3MXNpDohJeeos5xCNA8XBn6da511RcSdE1nT6TnUOdh5iJgRY3q7n3Cjf6tuKWz-pOCXk3y2VHP2soxrSFj0UUCyhlvay6ENIo9vB1wa8JYE9lRlGSclAKKx8a6jApSYOcRj7h5AQ2gcRTeP9OEqQhL7TN0QqaEutffSgcSsErBsf-bQEh9MLx9opJhWimPeYbm9YbXB0ojnbdjXQv7i0QbtwTwKIS0zmnThMb_8ogYm7Td8gJq3Z-sBQhlExZucjcFXquhxFP6t6wAxSGMLnkwk5KpgEfTP6fYCsh_tzzofIb1uBZgXh_kK24-3DvuI0ZmRaJFtYptw28q5Ep36Si9I0jvAeJrwlhEkDNgLn3WZFWiwF4Nb4IVV28z9vFNDZmA9GvhxD2PWpL7n6hMNnfg-cH2W6c1IbErYwjZVJ62hPvjpH0WysJHRMe5QLbWaVSOyd6nMoQ-zIHf6UbX6RkA=w800-h400-no)
        
        Graph đại diện cho phép tính <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\bg_white&space;\small&space;(x&space;&plus;&space;y)&space;*&space;z" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\bg_white&space;\small&space;(x&space;&plus;&space;y)&space;*&space;z" title="\small (x + y) * z" /></a>
        - Khởi tạo Graph trên TensorFlow
        
            Một chú ý **quan trọng** khi sử dụng TensorFlow: Chúng ta phải **khởi tạo đầy đủ Graph** trước khi đổ giá trị của các node(Tensor) vào trong Graph. Điều này sẽ làm bạn đôi khi không quen lúc mới bắt đầu.
            - tf.Graph
            
                Hầu hết các ứng dụng TensorFlow (TF) đều được khởi chạy bằng việc tạo ra Computation Graph như đã nhắc ở bên trên. Trong quá trình này, TF sẽ tạo ra đối tượng.
                
                Xem kỹ thêm về TensorFlow Graph [tại đây](https://www.tensorflow.org/guide/graphs)
                
                - Gọi hàm `tf.constant(42.0)` sẽ tạo ra một đối tượng `tf.Operation` và thêm nó vào Graph đồng thời trả về một đối tượng `tf.Tensor` mang giá trị `42.0`.
                - Gọi hàm `tf.matmul(x, y)` sẽ tạo ra một đối tượng `tf.Operation` và thêm nó vào Graph đồng thời trả về một đối tượng `tf.Tensor` mang giá trị tích của phép nhân này.
                - Gọi hàm `tf.train.Optimizer.minimize` sẽ thêm các phương thức và tensors vào Graph để phục vụ cho việc tính đạo hàm riêng và trả về một đối tượng `tf.Operation` mà khi chạy Graph sẽ thực hiện việc tính đạo hàm riêng trên các biến cho trước.
                
                
            - Phép cộng
            
                ![TensorFlow Addition](https://lh3.googleusercontent.com/2ILyxtn9E-UF3vtj_SqMaCOlQlvShHwGk0HxrzTDJE81GsjPPAG-9CW3nHxXO6Zl3QQ9PoVT0iRZGY4mQ5lkIEGIQjYCceGsQoIVIxN1p-w4AL7dtXcAW4m4QYirlivPe5yjL-RN7xOBrMmAa00TPLMMQv21XYWRv7yzNYpKq11DV9JWRJmbm9f8XtXQAl5d21gkxkQ63oJUsls4jS4zXQxidyQqcetrhSAW3QremRolGn8Ow1YCmHC9O7btf-Ze7UqznQhAhrn3wnlETATQLmQ84T7G6kgrD0iQkt77WLTsUXzPFi0WpoSKZlp5DRhKpUaHVKcMwz0mrBpuA_ivoNIdzLM76MnuUEfL3ELMk_Vag7gCKfCgvBpTD7MmnwWVuVmyIgLWH3TGtDWfuscxTSIgTPW8V0g15LjDwI5kj1LwJKypUj1UtiphXnx1Iuo6x8kg72F5c6dS-hQ67IUGtzQag3cATw8iffF9bwtJwX3awehgfJSO0wcbOuMah_WUEHiySQk9oKcFnXe_buLxJe2AegewdHl30tcm7MPSRtHCNx_SjePtoAR-ubKDVU6cWHhKy1yAHp7KCFK87jucpKD-TZszPMBKHPVzfWK41jHJxdfql2q2xgA5A6KU1Isl7fWSbBpX66ACFj5rKqazKzghA8cAZTSEHtryZCx0NFntv95wsswAg-b3Eg=w400-h300-no)
                
                Graph đại diện cho phép cộng: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\bg_white&space;\small&space;(x&space;&plus;&space;y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\bg_white&space;\small&space;(x&space;&plus;&space;y)" title="\bg_white \small (x + y)" /></a>
            
                - [Phép cộng số nguyên là hằng số](https://github.com/bangoc123/learn-machine-learning-in-two-months/tree/master/tensorflow/constant_addition.ipynb)
                - [Phép cộng số nguyên là biến số](https://github.com/bangoc123/learn-machine-learning-in-two-months/tree/master/tensorflow/variable_addition.ipynb)
                
                <img src="https://lh3.googleusercontent.com/EY44x8MIOT8lfDH0uJKXDfA9gmSqN9ICVEjlYhnX5X3DlBJdDMEzqtyI7Hh_mtyjhhMW_4pIX3ys8RjLVzPn2WbLUFn9s-gZqncinJqU_Wh8kvJE_3PMqGOlwdvSOrhAjFyRDbIQzft28jLu4sxqjfhS8UEc_vcaKgYSRosGLc4Si6cPlrUdrti9JoGpCPj_2oWp2UAd0qKI6p-uYD_UeWVmobGNmTEIO77-aFmiUmijcJ-1daDiIqWEhU-Qg3URRAScdrUK9erl5RlNn9JKQhk9zXvJg1wyzt_JaiQC_JiAa7t9hYj447kaxrSSH6h0H68lMopWbz0jMFw5gmgXFJ-NlXVakyCCCu8DNEqot-Id4qShXsk3cGBqtcSsoRz9kC2Asn5PnKSPM5bfIIH3vV2VzonrxlLu4dDXkQSaTdkNhuOi-BglmeruQadB59HzSmxksAcmMchnaJ2uGdf2QQ4v7yk_kpkDqeleqxsAfKmKPTOeKk59vu_GglBEs3u4ceD5_h09208gCJ8aQLVGY7W7ZMi7_8QqV9gKF4Klq8ZJDgkTywp5IATvkiZIQ1mayCECeFU2Rg0BZtqh6JploxAq9KHYtN79FkpuB8lBpBR8_6o0ujDga0-Vfgp3GLRkCsDYzimaoOdmlej4iDY3JRBMb408aZBMs6OwT_utXHeZkeE6Mn0Oug7M=w800-h556-no" width="600">
                
                - [Phép cộng ma trận là hằng số](https://github.com/bangoc123/learn-machine-learning-in-two-months/tree/master/tensorflow/matrix_summation.ipynb)

        - **[Quan trọng]** Tại sao cần phải sử dụng Computational Graph?
        
            Như đã đề cập ở bên trên, **Computational Graph** hay **Dataflow Graph** dùng để biểu diễn các phép tính toán. Bạn sẽ phải tạo Graph trước rồi sử dụng TensorFlow Session để chạy các phần của Graph này.
         
            ![](https://lh3.googleusercontent.com/NIS0mI7clNb6y7twlSgpak8d9CnVsz0SAMJaQX_xqXZNFnw3kFtXvstzEkWEtmSCj7W3I2APu4bXXyqzAFoGnvvP4dQfmiTWQbqOQIZKvLZtSJG0sRnT0Sb17G3_OD8McibcHRhtWKKCXFzXQGv08HLzQCf5f4S0mPyLjQFw8REHlI_YDGQUTz1VRqxCOpV7cVUKA45kSvSII6LCB0o5JmpuzxiIl7TiQewF4mUGODHeRLEtb2AESSwnc7KOZYjySZx0HeFRI4B3jc_VxzuARBN6Phc2TgIUbeoPpWN7VLX1FeBnGXPQimwaDyJ9Rbx8wOm3-USKtUzwUy1h7ZYuVCmd3zORQg7M3dJvEEbOd1CjyEqqkdzipwJ_Z2_d9wJmW29dAiWUf10dcjoG4mCJKPanNBH9nA4Dc4Kxd7JrksDpRR59S7DdB01IYjXjXiCt5Ol3KG_Fja_1EkvrTjBzlvkcU3djSAOPZCHaf4hfa52jXvcniB4HudneJjC_j6r_Q39iRghaiVt4Na1A-flelpjCtdIR71Jo_Qh7ZKn5gbe-V6x-l9Frot7nIIbBkacNWx1hB9ty1Dt8f-phxkui_aaOHwbp2SPVsrPFm0vgGrO4xxbXONfGfDhZ2Qy3gtiloyKiS-38TwAZf80iLZDeQNdk45y9qneexDAPkPEp498uW3FidOgXneWpPw=w252-h448-no)
            
            Dataflow là mô hình lập trình (programming model) chuyên dùng cho tính toán song song. Xem thêm tại [đây](https://en.wikipedia.org/wiki/Dataflow_programming). Ở trong Graph, các đỉnh (nodes) đại diện cho các phép tính, và các cạnh (edges) đại diện cho dữ liệu đầu vào và đầu ra của các phép tính trên.
            
            Một số ưu điểm của Dataflow mà TensorFlow (TF) có thể tận dụng được:
            
            - **Tính toán song song (Parallelism)**: Bằng việc chia tách các phép tính toán độc lập, TF có khả năng lựa chọn thực hiện song song các phép tính cụ thể.
            
            - **Thực thi phân tán (Distributed Execution)**: TF có khả năng chia nhỏ ứng dụng của bạn cho nhiều CPU, TPU, GPU khác nhau trên nhiều máy khác nhau.
            
            - **Biên dịch (Compliation)**: [XLA](https://www.tensorflow.org/performance/xla/) của TF có thể dùng graph để generate ra code chạy nhanh hơn thông thường.
            
            - **Tính di động (Portability)**: Graph là ngôn ngữ biểu diễn độc lập (language-independent representation) với model của bạn. Bạn có thể build một Graph bằng **Python**, save model, store lại bằng C++.
              
### 4. Bài toán hồi quy (Regression)
----

   - **a. Mô hình Hồi quy tuyến tính (Linear Regression Model)**
        - [Bài toán đơn giản](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/models/linear-regression/README.MD/#1-b%C3%A0i-to%C3%A1n-%C4%91%C6%A1n-gi%E1%BA%A3n)
        - Định nghĩa
        - Mô phỏng thực tế
        
            ![](https://lh6.googleusercontent.com/xQMPlPjv845-gDphAAcyNLJufG_774prU974Zn00WP8Hsfandysppi3OiizibH_fdiYhODkhrmTKOc_HtoC74HHb6s6L0Bj4n1jv_P-xJWJTcSDgsO5ncISfO_SKLGq-8W3o2lVgf30 )
        
   - **b. Hàm mất mát (Loss Function)**
   
        - Định nghĩa
        
        Hàm mất mát trả về một số không âm thể hiện mức độ chênh lệch giữa giá trị mà model của chúng ta **dự đoán** và giá trị **thực tế**.
        
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
   Nếu bạn chưa có kiến thức về thị giác máy tính (Computer Vision) thì bạn nên đọc thêm về [loạt bài viết](https://github.com/bangoc123/learn-machine-learning-in-two-months/tree/master/computer-vision) tại đây.
   
   Một [slide](http://bit.ly/NextDN2018) mình đã từng chia sẻ tại sự kiện Google Cloud Next MienTrung 2018.

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
   - **d. Bộ lọc**
   
        Vector/ Ma trận trượt trên ma trận khác để trích xuất các dữ liệu đặc trưng, loại bỏ những dữ liệu không cần thiết.
        
        Bộ lọc có rất nhiều tên khác nhau:
        
        - Kernel
        - Filter
        - Window
        - Mask
        - Filter Mask
        
        <img src="https://lh3.googleusercontent.com/tNNA3uwGXN_ZIdsX597nWgrQYUk0E3A4FAxeVThdHXArmzzej2RbVD6aCSfj1nQZoXQuP2PRa3y8uMYDnlS95t577zb3g1GL0mpjAonyYci0CbEOib4K_uVdDwlK_Nzy7BCY1JodZwpzTihb-sCTweRCe1U4vv3u51wfQk7k-giQ28RxpI-a8KEK07S_l5TGxdz0-obHTj-Scybd-55nbwevbgMtDP5O_QzITvGbIbxAC2U5kcMH6xHd4mCKQAQFuSC3G-FyhsJijUz3n4E-AavavQDqS6bKqkpPORa8Sx_uOINP9UTp_TlotRJXSstNEm8jslBaQWJHbKdqJ2toSP5t2PKQsJRONJ_M-Kfw-JpxWiawyl_yfHYOY0pR8hPYyaZKXnDviEU4SkPtlFCVUshhDMLvxaBW5GsDGDHe_OsIemQWOq-TXcUieeR-6LIpSfZVc55wUHKZ2jQPO550K9v4kREAsSklwrQDet5q5ljNhgwWrsu0hUxYdftFGACMfxPWMuq4E1k4mWyLPq538TDP63M2mVagG7zgwQ_vxKlgz5iB9-3MkYTbrfkbtQAXAnhXm-Sb1WLHNy7Kh7GB5ni9a18MKZvAMKgR56yG8jWq9O9TjQOt_1JufTXkfdc056ijvCpEfRiUgxPWR5Xa5MMPRE1zom1RLOh7-Iw9a0oqh8jfpK0FPACnUg=w495-h542-no" >
       
        
   - **e. Phép tích chập (Convolution)**
        - Công thức
        
            ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/271b9c48087f01b626627a423d8e9003aa765728)

           Mô phỏng công thức:

            <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/Convolution_of_box_signal_with_itself2.gif" width="300"/> 

            Hàm **f(m)** đại diện là ô vuông màu xanh và Hàm **g(n-m)** đại diện cho ô vuông màu đỏ. Tích chập **(f*g)[n]** thể hiện mức độ giao hoán của hai hàm **f(m)** và **g(n-m)** tại ví trí n.
            - Tại vị trí n mà 2 hàm giao hoán mạnh nhất, tức là 2 ô vuông trùng nhau thì giá trị tích chập là lớn nhất
            - Tại vị trí n mà 2 hàm không giao hoán, tức là 2 ô vuông không trùng nhau thì giá trị tích chập là bằng 0
            - Giá trị tích chập tăng dần tỉ lệ thuận với mức độ giao hoán của 2 hàm
            
            Sử dụng tích chập, ta có thể trích xuất tính chất từ ma trận của ảnh dựa trên những ma trận nhỏ (filter) cho sẵn. 
        
        - Tích chập 1D rời rạc
        
            Tích chập 1D là một vector (filter) trượt trên một ma trận, qua mỗi bước trả về tích vô hướng của filter với giá trị của ma trận mà filter trượt lên trên.  
        
        <img src="https://lh3.googleusercontent.com/0gjsMXWHkPKNIgKT_5Nj4j3K56b7KimH9htGFoU9QIELEFEDuDD5Wx3U7aCWuiV_ol8O1NjxqLciL_9earZ9u17z-o-eNIoSN-bRXm5owTwmVJAZEueN_NYYDaSqoqh6tKEAsiQlRTenX4kcNpRJIV97IB7CRsIsjTA8ISeYub3wpwtFRdIpcPr24d0fxOwU4f7KDixTuPJPTqTEGMy4fxrOq5XYBldfPLCTjzV2Vvdl-1ss7MR0m7sB5c4ICf5ceR1aKago6E3PX_dkP5fSpjPzenDNGB3Uj0O00a-foO3IFYwg-cv836azuAkXWfWsRABP6S-V8iUySeqF_oOyMx4yTmISXhPIXOpSRGDe0ru4P2leDbDZRM7sQwwhdm2XxHXfUU798acFKYEx8xZvgy8n8mncvhbcemiRfocDFMvBhnVfxRrMdyvtmp1c3HNgXRi4U-tI36sR_ByvHMK1Ak7WYEuW9wXweMgVtAZrXdADXcdf9nB2CwOWSNoPh83_VCz1bHD-5VyeKNmto50MJ6igmYxRvw84Ghl2UgbSqksMCMgVfp0UfhvWEOJpwmo57J54NYIAZTNn0NOPdAL6Y9a0_BM1NT1oGjf69q8V6u8r5SLmxbInBD7mJA-ZSelwxWE4AQiJ8IQPTA_OKmRJXcAxc1G5B7IFM9gvvXfR6UW-I8Bt5QB9Wdq0tg=w800-h447-no" width="450">
        
        - Tích chập 2D rời rạc
        
            Tích chập 2D là một ma trận 2 chiều (filter) trượt trên một ma trận khác, qua mỗi bước trả về tích vô hướng của filter với giá trị của ma trận mà filter trượt lên trên.  
        
        <img src="https://lh3.googleusercontent.com/6-CEfDinkmL90ow5FJMmn2OyoKJ-6GgybY2zglGW120mlPouza63SZPGkM-IyUGWcBRQ1gjKkRFm3KKdRxVyD-YOwpCYxvXqQZQqGxkFDzP09hPEyT2xfkoST-h5JpMrSw-tKZR-wtKoT8m9APgyqYr-9kYomGXD5Bjaw8vltsx2bszIeJtfmFnBDNQmUYTtmhOriI2nWmsRB8WR9wirPQ99to84-OfMyqZrFiZjOlPJo3ETBNOHTarLq9spYosfqCNlH4UhfGeN9A0zeo_YeoBus7hICEWe4_8kFnno9sDpori9CyXnJCL7SpS4bh4P4CDcKoKuNTnK0bMp4B5pMyrlLaRZdut6E8XtIHpUenJ6yhl-Kx6U3Mj6C0RU0ASEByyeWk5bdUtgwTBlqDQSRBmKNZB8CY-cNVW4IvG6EoQeVOyb8z22A2A66Cfx5rXV47pI5nJ7ZWlVj9Hmj572m-VPJE-6HKW0h5FxcKx-kdMS6F0cPK3BBUL33TjfgSEwz0Z_LBILhumBQE0aQC6W30YG8XAxiDPwIMWH7W6UuUnAOyIMx92MRcaXMTDX4YjPj1TeJZm_SixN5JXguOCs0piTZ2Uhp7ym6naY_YGdNyDb2AjIn2BUSvCiLzxLnRMza8To7lTl5xA-dIErVk33VrjM8PnpjeujTX0IN3k81cEZhvVYRqj7JnWRhA=w1280-h720-no" width="450">
        
        Ảnh từ [machinelearninguru](http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html)
        
        Hai cặp filter mỗi cặp 3 filter trượt lần lượt trên một bức ảnh 3 channels.
        
        <img src="https://lh3.googleusercontent.com/eH6gUF2CmDlfubKj68ulkbV1nv1JSW9VF_HRukiMULCePzMeBG9cOHYHtPvy5GToWNvYBVdIvONAJNZrrOAjt7Kzd6qIfUH7doVnQcN4pqsdnczGCGkz-tiXfJzTDgOAJ1ZhAaxywMX9PON-2JHLRyWrYBR1fx4lvNYdF5MQLOLbFF5wcj1Uhco7xAnmQWbWi_tEji67XvTW5qJpN_sfWji5Uze8qJ2WLtNq88_yxxrMEQvf1lq2K7UHprYpT9g0I1GbcWPXpzbV93b8ywZ7oNDP9FqwEMKE2ohhfL5zmE-VGkNnlLl2AFLl4HyCVtOJjnoxESVyxdkC_aCqQENX8c8r6l3RNj4I8vrwNn3JKgiE7Mq0clL2f9nxrD46mDwaYcEjkQgXVBcJNRZG5CVLmBp4pktjKh5bzhWSMnt0UYXMOVwoJ-XBfQrAxfFRns037oNVG0JzwiT1VdKv_cgTpUeXqpHD-39Y2_bX1b9X39bIJTiRwC_bfSz7wBVEZEeef7s8zOVJ7TDM2OAsmnw6OBdecuxjf_UYyVZLADqplcyOMrXvGhJHnkuxSuNye6wi6dZKIuPvwFHNE21FLTTzThxVBBaeDLAcc3ozrBu6h3DhTWOz3Rcx7nT7ooTaxveMhZ5FLuekRJK1JeLK-Ga65zr-0u3KS9HU74JrlpGyoH-F8N-HqnwNSsHeSA=w512-h424-no" width="450">
        
        - Loại tích chập
        
        Giá trị ma trận của bộ lọc quyết định đầu ra của phép tích chập.
        
        - Bộ lọc Sobel X lọc ra tính chất đặc trưng theo chiều dọc
        - Bộ loc Sobel Y lọc ra tính chất đặc trưng theo chiều ngang.
        
        <img src="https://lh3.googleusercontent.com/NDrrNpzg6eJS5J436KId9XzytYiD0ImIuoBi4axjBK-SCoJ8KYbd9xu7aZsbBuU8DT9qDIH5sy2Tx31Tn2O6ZKtaaNYEZjG2bjjxmj-fcVr8AQVIiuaYaNU9F6Tyo_s1JSmwqEa55cNTvQMnaNfPyMAzIXh160LN5PMzUHy9rVqqlLnO3r4i5Z5neid75rk6Qxrz6bQz5A1TbDgCAZZgRasymhkiRsCqz6-j8ZldsSNO0RGIB3uEzo8ghXYUpeddQaDxoUHZpHYFhF7EDvbprt09kvUw4VGchB-LswMbpxoOElJ7Mjo8eEVmcJtUKd__wimSkDaP1LyZFdo0YocbTfKNM8hn2blWwfW26NazgMIfe16UF3JmfJRwg0Z4dlUamarYQOgTIdH0GXTlAdG6TVM-rM2yG4XzgDzDSfUTm1nooQwZ5a0FTOL5gAIR1yCbTvW_Lm6YtfRqXkPEdnXx6WMd9bj9Y4VNdLAZ8Xxmt36q9dFSwO7GuCq8PCbWhF7qbobMUtFBbb0AVjhakDn615DhXni66tfH58PRdfe_9j2aEbpzffy-ZhEPZ2PcHVFE8EKSgABtLKOTQz4CQKX9YJG2WjjQuu92Q6vynCEmzgKC6DurqQcZopZEPxqWmhI6I6XAq9PhqRBxK1ZubRYbjQ6FSPA5oClxNuhQNXDR0MRs9UtShMDC5lf0NQ=w1932-h1106-no">
        
   - **f. Pooling**
        - Max-Pooling
        
            Là một dạng filter trả về giá trị lớn nhất của phần tử trong ma trận mà filter đang trượt qua.
            
            <img src="https://lh3.googleusercontent.com/gfOoJDvIeiuJEmkGP3zeAl1qhSIR2CTIxwJscXvGUzE-JBg-ignRwrDpGXHG6bQ6HbA8Yy2hvE9HmAl1yj_3Kg31kL0l1FHXzCAl1Ws1OQ9Ted_Mxn93a4f12eqS-KniWFV27iigEw0OWjTS7JO4Ci_AuH0GfBVd-8Zht7MhEsZxmoEXI6W62m8JO7i1793CPbUyMN6SEf7OaJhZBif5APBA-Ro0zTN40dg2Y9vGlpZGtO-Ttu-LhVVcryxR_B6R98LNZcqBoRUFZ-2oSPr-4OH3F522X9HOFC7XlndhNOi3DxpUQtyLgtBgwQ1XjRTmdathKEgMhmPJ0GL7B_pBIIrODgseh5-NKz1ftuJE6bTj6Qg1urV5kW6TCxcHTLPVtJG_EYqbi8O-SxEmXwsuq5H7SIyHx1vlC2SrTkNLLN_bk0A1w1GECkaV9yd3-Y13t-cNRxkyf5nuEvQfy5QAeTq5z_2UNqkEeHGkKbagRQJEtAV95tYu4rUNbVbO-i4LG0mamKXj3qm7jIrUrPlAhYBi4-LNfGYUZvH69_z9t5lZMszETUuO-XPjbj0dtolteGSJsPGrGwO1cAUabY51EgxuaIW7X3H0im-uAagfC67RDzIRcYPIMzQXWsgaVZa0fkQMgVqpb-uRo5B2FxatUie8toWS6LD78r__cfABqhF4n3x5KAOaWh6TVA=w728-h324-no" width="400">
        
        - Average-Pooling
        
            Là một dạng filter trả về giá trị trung bình của các phần tử trong ma trận mà filter đang trượt qua.
        
            <img src="https://lh3.googleusercontent.com/_p3jKDuyhvhwvdu29SucUNRv3MWBD-8XXDFwE47LootNPAb0Mi7ZZjVbh20y2oDWjgJRckmMQxhe-DHuwY7rxMPxnJocsz7RyBldGXRsPDrIlAl6rt1YFWXn1L-laO9u8S0j0S2PIFFp2D9pQZ6AvMuk9vrFfq8njrI2ZqNQNHB_U90uRbHinDDPmbDlU4fesrlkNpoWtyBMkg1k-Cx0lHXtdFn9ApHjBm1WKjD4ONX1ZLOnOFBney4BLSjrZEP5PUIsD7yPOinraZ-gDnIfNlaN70se8pXitXXYqygL7X2FZwywOzV9jvE1DjqHYM_UYZhC2q4ESMtlyvtA_MhWgYMgxZKay0aYJSXafXKxnFkw6dbPngPDpmc0nCnTjjiMLkVp09Jx_fFdte7ntaW3OM4d4KNLBWaOc7bUt8YVlrXSVl0M3ZVxdWgxNqSFl-in8MWODuN0nDVyUiooG_369xzMTrAvZg79S1BRCZTiQc9c_nhg-F9vdeqjLqAESjhbO5DmxFUu3fO2AVls5LwmsuEDS66H-ty9URYYe7vz8K9wHiXll_JVvdGz7RUWnjoWST_zn2yBgk3NSWziSX_CKIQUuyHZFsOQswOWOTrf19L_75QAZwY23v9NZUu9hGA4Bx75VF3psSYTrllBT8ogvoXyryJ_-pVF0WTsJS2HyMPoXqIksBqD5AHykQ=w318-h149-no" width="400">
   - **g. Các hàm kích hoạt**
        - Sigmoid
        - Tanh
        - ReLU
        
   - **h. Cấu trúc mạng**
        
        <img src="http://cs231n.github.io/assets/cnn/convnet.jpeg" width="440">
        
        
        Ảnh từ [cs231 - Stanford](http://cs231n.github.io/convolutional-networks/#overview)
        
        <img src="https://lh3.googleusercontent.com/LXq-jmPcdHnxHZy54FsOZzgp0iEkTWylLpxR5YAc3kxlwbNOx-7A6Z2CxwZW02g8WPp2rN8XBPIZIArUjVlzpba6M9skHzMq_H1rQoxgCSg7BuGYQHEZTmxfFyLgPhKPJzT8QE1dZWWTQeduCRY5QAbwGwQy189mfr_aU4XE3Y2NiTJpVslqEwt-X4T5CvSjI8lu8y5BiBjZm-eHFWvLAX7l0m8McT6S3ll-dOPSQO09Dl4POExvunoWJy0zJpFMUaKsTSiHmsTdtKaq5Htcn-3IV5600hluLP6BB61kQspe7acoDrKMmytqcc7UiAJvPzN2KoH1sA5i9D35XCjbX1mtDkbpfydHbnyVLSO71JCI2mQJc-BRv_2vQYEtlXtrK73Y-uG2PQ0L_kODEWt2EDN2cuqfxYflbDkT-cGKm7zJhjCtfdQV4ivdxG7XbjUvJhWuzvGQfn-9GVLjLI8CsOjOiKdsRhtIjq3pLgt0cnnKnurna_BhihPxGzzzzwc-Qw35-obDXreac0wHZtlqMNuHLuBT7bqEI2q2USUd7KlgyaxS2XfrQpvcJti1TMeV64G3ExTKd-XTds8nP0UAK-0LNq0p9bdhhoIJCBEOLsfiCpUJsd153gaPnbnFWUHo0chKYYNd3FtU0pXzQghrojhPdBq-LnQ-umxv9W7xzCCW58YuOFvEeEqf=w748-h178-no" width="440">
        
        Các thành phần của một CNN
        
        - Các lớp Convolution (Convolution Layer)
        - Các hàm kích hoạt (Activation Function)
        - Các lớp pooling
        - Các lớp fully-connected
        
   - **i. Công thức tính toán đầu ra ở các lớp**
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

   
