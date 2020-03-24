TensorFlow là một thư viện mã nguồn mở dùng để training/inference machine learning model có thể sử dụng cho cả nghiên cứu và sản phẩm.
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
    - Hidden Layer của một Neural network

        ![](https://lh3.googleusercontent.com/DQRiptuvHRUOG4QhqluO4qeTvxTHSmLqifioijNjCKx9vVPYqueiGMF8S_Hr9NLQWImmDncT7sI_t5glNo8WM2McFavdjhTKcLmylMu2x0wsNjRwuD-1ly-qRyr3qsZM3HN0V1ugELymzQ1lntvEFzIeEcTCUDHQgJrff_26YSjDsJz8YL8aACH8_r9uz6Szk57wGAqHA9SK36YIsUomHo9TQl696MKt60PYzWIS13mmalIXURQVIDxhxv38rkxPc29viNezQytwSKyXVKpN44ijCFFUeVPHz3iytv8PvOswtOC38LS30PFmWS5F6BjBAk5C37r_xC_PU-NaYPq4wYwnFy3SBnJYZZmxli-BTP5tvhSZlbxTrpC3lgjAIjNb1Y5AwmOjNwd9VGKW2lXlB33iK5X927dX_FVYJ_ytShppg1BthE77BtY2R7Cw_h8W9jMTiL9sT_7pFxCRhcjArcbr6iucP_jPJzA1AfePsVuiq8X8LmFIZKb-NWpSV_pGha_0SwacnzN3QHstmzz9cc7Q__1VjU0TnVnjtXE_LLcxNXQ3MhFNLlaNv4MEbwfJGnJjmMuGUhlgXP3_0ALshAjinkUPIlAoZ9PyzZeX4p_zY7jkwLeFWFtWgK9FwZIf61Wgx8XGaqlLjIWNLjsaVj42dtGU_SkplJ6b4ci_P2_vupm7tkiK0FJ8=w800-h400-no)

        Công thức: <img src="https://latex.codecogs.com/svg.latex?\inline&space;h&space;=&space;ReLU(Wx&plus;b)" title="h = ReLU(Wx+b)" />

        Giải thích các phép tính toán trên Graph này:

        - MatMul <-> Wx: Nhân vô hướng 2 ma trận
        - Add <-> Wx + b: Cộng ma trận
        - ReLU hàm phi tuyến. Bạn chỉ cần hiểu đơn giản về hàm này: Hàm này sẽ trả về giá trị của **chính đầu vào** nếu **đầu vào lớn hơn 0** và **trả về 0** nếu **đầu vào nhỏ hơn 0**. Bạn có thể hiểu rõ tính chất của hàm này thông qua đồ thị của nó:

            ![](https://lh3.googleusercontent.com/kSJRprcz8A9lPXNPb3h8kuWG1EKue9ck4_6zirLRS_2RYYKn71NR4OF5kJctorcOsbw6Px0CQz9rncI-LNsOetQduSjIZh23R454Y1pe41FVxaFyB2prcbk2BP3pbgEnABIwpB-16ayK5QwOAbWmjA7khdPhl1T29dO46BfRiC0h8OS7nj1NrpVEFjQUiKN6xdS0O_7jkpLyx3EAF6AHiXRy2Ex_iFby86PSVJcMygHjpNua9VWeXkOlNzI4bKRc2_VJEDsHvCgvpeOLzJp_OdpioXhC6Orbe4fkILQRzNsZhbCM0cQoAYtn3lAqWz1qxNmA0K1egan2YsfEIFMP8pRCU5Z_aQK1hMJb4g6Y2PtQA3X4Cr3T0xYyyK2BWXw1-Dox_EtVXL2Mu136jAfoeCa8g26FTuMlBZ0VJqwfvwUpAsDKI3qMSe9sgjGqQohuwbsPd-Z1qjNWJMdybFaMGmhZf5E0TuXQEBzpoZ3T9wCjxXMOBfLb6D4J4wIhdhyrX-7vL-3XeeFCY0VOMY6ZbUndXW_kD0EVaXdyYVyyHNm8-OJD5vXOyNRI677IuzjWrsKjup0sgA2nQRdnyeq1Eg7M-x-IS8P_4RpaEyT9FRKEW0zJ0GawCBwbg4BlUDkXhrvuoR6TIfktZq_2h_XnYTJ-Ob9tAVNP-fUuuYSear3iKKzZRADY3-dY=w311-h210-no)

        - [Tạo một lớp layer ẩn với TensorFlow](https://github.com/bangoc123/learn-machine-learning-in-two-months/tree/master/tensorflow/hidden_layer.ipynb)

            ![](https://lh3.googleusercontent.com/2c4meujxbGKri8GvAdlCQAOpJwzQz4OY9yoI6kXsHk7g6yDWNYyIh-YzrMCmJ7RCvzNtMuhrF28w2gndh020D6m3KURuPkOAEbVtNTmqlgzJA_eYzMMzYu8AATZQkcN6mG_ioYwYISVtlSXCJUaXbO93cncQ1lv4168WjchUi47XEqbk2-8WfcDitU9ZRpzWJI4ofbFMlGzlB4d_VfJYZeWnQec6xPWPDRxtkG4D65YGY8kLSBb5dTjU6azwUyItFdp2rhNRyNi36w6y6h8E5_91KXKkQSaAjbhk_IfrhIEVstccyt3Mjj_YdVX5vfbmbiqpL8me-o5ynpYEceEcR0vHLQKccktp5dmVdvXePln0pSVjm3RLyWaKydE8aifUKJd9OFxQilqbwx_SxrF6XX2E7IvcWt5Qi9ZTKh6XP9b8BtVkFfKsU8VCHsR6OtIxf7J4P0bQZVzRdd_qsVZMo2JrpLHKirbKs9hLl7XqQUvrHo0lFgJOq004p3N33-fe3VMrDb9Lz7A6CMh_OU4k05VBow3By6Cp3XNFxIifD6kDXcEBZQgZpDlJrdV0YOef0hEBcvcvrgM1iflrGaPZibJVQM2iCkKvsrUgfMpGpW8zWzI9zXKZv6S6MjQPRFL8P30Z9CMy9yQy1URKJmc065Bt7XHi8qhItAvOfGafRc5fcxe4CuuXWCNg=w800-h400-no)
                

  - **[Quan trọng]** Tại sao cần phải sử dụng Computational Graph?

      Như đã đề cập ở bên trên, **Computational Graph** hay **Dataflow Graph** dùng để biểu diễn các phép tính toán. Bạn sẽ phải tạo Graph trước rồi sử dụng TensorFlow Session để chạy các phần của Graph này.

      ![](https://lh3.googleusercontent.com/NIS0mI7clNb6y7twlSgpak8d9CnVsz0SAMJaQX_xqXZNFnw3kFtXvstzEkWEtmSCj7W3I2APu4bXXyqzAFoGnvvP4dQfmiTWQbqOQIZKvLZtSJG0sRnT0Sb17G3_OD8McibcHRhtWKKCXFzXQGv08HLzQCf5f4S0mPyLjQFw8REHlI_YDGQUTz1VRqxCOpV7cVUKA45kSvSII6LCB0o5JmpuzxiIl7TiQewF4mUGODHeRLEtb2AESSwnc7KOZYjySZx0HeFRI4B3jc_VxzuARBN6Phc2TgIUbeoPpWN7VLX1FeBnGXPQimwaDyJ9Rbx8wOm3-USKtUzwUy1h7ZYuVCmd3zORQg7M3dJvEEbOd1CjyEqqkdzipwJ_Z2_d9wJmW29dAiWUf10dcjoG4mCJKPanNBH9nA4Dc4Kxd7JrksDpRR59S7DdB01IYjXjXiCt5Ol3KG_Fja_1EkvrTjBzlvkcU3djSAOPZCHaf4hfa52jXvcniB4HudneJjC_j6r_Q39iRghaiVt4Na1A-flelpjCtdIR71Jo_Qh7ZKn5gbe-V6x-l9Frot7nIIbBkacNWx1hB9ty1Dt8f-phxkui_aaOHwbp2SPVsrPFm0vgGrO4xxbXONfGfDhZ2Qy3gtiloyKiS-38TwAZf80iLZDeQNdk45y9qneexDAPkPEp498uW3FidOgXneWpPw=w252-h448-no)

      Dataflow là mô hình lập trình (programming model) chuyên dùng cho tính toán song song. Xem thêm tại [đây](https://en.wikipedia.org/wiki/Dataflow_programming). Ở trong Graph, các đỉnh (nodes) đại diện cho các phép tính, và các cạnh (edges) đại diện cho dữ liệu đầu vào và đầu ra của các phép tính trên.

      Một số ưu điểm của Dataflow mà TensorFlow (TF) có thể tận dụng được:

      - **Tính toán song song (Parallelism)**: Bằng việc chia tách các phép tính toán độc lập, TF có khả năng lựa chọn thực hiện song song các phép tính cụ thể.

      - **Thực thi phân tán (Distributed Execution)**: TF có khả năng chia nhỏ ứng dụng của bạn cho nhiều CPU, TPU, GPU khác nhau trên nhiều máy khác nhau.

      - **Biên dịch (Compliation)**: [XLA](https://www.tensorflow.org/performance/xla/) của TF có thể dùng graph để generate ra code chạy nhanh hơn thông thường.

      - **Tính di động (Portability)**: Graph là ngôn ngữ biểu diễn độc lập (language-independent representation) với model của bạn. Bạn có thể build một Graph bằng **Python**, save model, store lại bằng C++.[](https://lh3.googleusercontent.com/2c4meujxbGKri8GvAdlCQAOpJwzQz4OY9yoI6kXsHk7g6yDWNYyIh-YzrMCmJ7RCvzNtMuhrF28w2gndh020D6m3KURuPkOAEbVtNTmqlgzJA_eYzMMzYu8AATZQkcN6mG_ioYwYISVtlSXCJUaXbO93cncQ1lv4168WjchUi47XEqbk2-8WfcDitU9ZRpzWJI4ofbFMlGzlB4d_VfJYZeWnQec6xPWPDRxtkG4D65YGY8kLSBb5dTjU6azwUyItFdp2rhNRyNi36w6y6h8E5_91KXKkQSaAjbhk_IfrhIEVstccyt3Mjj_YdVX5vfbmbiqpL8me-o5ynpYEceEcR0vHLQKccktp5dmVdvXePln0pSVjm3RLyWaKydE8aifUKJd9OFxQilqbwx_SxrF6XX2E7IvcWt5Qi9ZTKh6XP9b8BtVkFfKsU8VCHsR6OtIxf7J4P0bQZVzRdd_qsVZMo2JrpLHKirbKs9hLl7XqQUvrHo0lFgJOq004p3N33-fe3VMrDb9Lz7A6CMh_OU4k05VBow3By6Cp3XNFxIifD6kDXcEBZQgZpDlJrdV0YOef0hEBcvcvrgM1iflrGaPZibJVQM2iCkKvsrUgfMpGpW8zWzI9zXKZv6S6MjQPRFL8P30Z9CMy9yQy1URKJmc065Bt7XHi8qhItAvOfGafRc5fcxe4CuuXWCNg=w800-h400-no)
                
