## Lộ trình học Machine Learning, Deep Learning cho người mới bắt đầu

Tôi đã từng học Machine Learning trong vòng 2 tháng và tôi tin bạn cũng có thể làm được.

**P/S:** Hãy để lại **1 star** để team có động lực sản xuất các phần tiếp theo và cũng đừng quên chia sẻ tới bạn bè của bạn.

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
