
### 5. Bài toán phân loại (Classification)
----

   - **a. Mô hình Hồi quy Logistic (Logistic Regression Model)**
   
        <img src="https://lh3.googleusercontent.com/qyA4iDLK6VBQ8LBID8QM12f9NokoRRzf3J5SsOl1ALG8lrtdvpt4QixEQ7Xcnt0BfwIJLnA4WribRmIfOrHXfxM-Cmt2ugrs3nNdvh_7GSGW8sTMvPdmCi6nCzGxH1Sy5TReNV4kL--o7tSGHhdsXrtSzrmDCK0PG_orWhBJS4_hEmvE8tujI_S1fYtebKnWCu1VvU1DN8Opw8wFIOaj1bQZkucPgJcnx8JJer9Wc8sfRhzXRTus5ELxBKw0_-eLWzsnvVHlNK5u60_-f_E_5r24xBhDKe4YRrFiyrU0P2zUIrW0d_VK5WYzZkJlYuzBQjxWMdUnuGK30mn4-yn5iTYSPxdvFXZF8uhYcmbu2739EcF5lTfTCIbwtOz3KiHIdPr_yZYTgoEP-qcD9zWB7SRnAQwtQiwe3lWdLFVlfDTU9Z8pb8Fz56I1YRE4kGni_L08tXl9ThnuzVjOm3x8d__E42zZDIrjxVX1XtHcnKruvJhdxvfxoVshJtB5_Eod4Ms0UhfsYv8Gpd5K0GamXS3iWYaeTdx1FQ87JMOqKhFWEfr1qTER2IOJgM8phziFB2gUzZJWuFnINyHUQUPlf0Djht8shTDcykzrnZ34HNw0bWG5ErWfY3xzPAhfE_mVi9MenOMG7hl0quRn7h7WWyNroWAnXfwEhhGS4lE0gQdZaln0Q8HKTu3yxST6pZnCW_FKmzoVXAdLx1Mgkw=w900-h500-no" width=600>
        
        Như trên hình chúng ta có 2 bài toán phân loại: **Phân loại 2 lớp (Binary Classification)** và **Phân loại nhiều lớp (Multiclass Classification)**. 
        
      - **Phân loại 2 lớp (Binary Classification)**

           - Mô phỏng           
           <img src="https://lh3.googleusercontent.com/A67h2lsxXgkXLWeNDoC2b9IzL5iGFegRDtBPdulR8H1JVybn0dKXyJHW0DVvyw_AkuAjQ4gwhJibAcg4tKgoj2ABLB_bXl6WfDFWkP8-3UXTmAdUYvmc7ug6DwO37_VoeKhbgwHYxqh9NNuKQbsF0l_DSMC4HwmZ4bEJEjSkDqMhiDxkbB6a0AYr51UzB49UUYDgviIfptVu9tGJV8sKxF7RcP9ByO_C9N5fK6lBSWmgQz00w30IJVDFL_1Fow47JjSiV8midSh_ptpqB54lnF2tFSidOAwUkGsiFhQYeXZSWiHF1iFpw01O5jRbsnJNlpKlR6L8EG3gNHZITDSNkoZRph2iZj4fsP9uyEWpEvBSRD0GCckZZxFqMM7YkrVi6kZclJwK8YQI0pnMiAAQy2LtAVMKitw-qusHU9DCxyJgpuXU7asYd879LXb8vEWnBV2_ouvPibinkxSpJMAN3bHz-9lMcDysWdGovcOQuVP7iXeoHmcTC9F7BBxlL-s2zqNSKy7VdY8Vw-XBbkWYz70Nq3ZX7xrjmcaXPaMiNsJF4p0SFayzoGfI1zaoOhhEApqdc5zM6fkaTfuOmfAjUZQUN88aqQOEHLyC8CHyk-8odzOqbAfldUxTTap79XSJVzAFm-ghWJA7Hm_y_oO9uxUGy8dv9UMmqGCJ-EEF9-QkoKucNO9wvVSQJtsQmsUiGKW8B6ZdwwwYG2wWaw=s800-no" width=400>

           Bài toán ở đây rất đơn giản, cho 1 dữ liệu bất kỳ có giá trị dựa theo trục <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X_1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;X_1" title="X_1" /></a> và <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X_2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;X_2" title="X_2" /></a>. Làm thế nào ta xác định được điểm đó là thuộc lớp **ô tròn màu đỏ** hay **ô vuông màu xanh**. Vậy chúng ta có thể kết luận đây và bài toán phân loại.
        
      - **Phân loại nhiều lớp (Multiclass Classification)**

           - Mô phỏng
           
           <img src="https://lh3.googleusercontent.com/7yNQ_CW4yCcDRKi8lT73ImAcsH-xqR6Ew4CGCeEyfwwY5960y557GeN7mWatO8h-6P8TS0qWd3BarekpYscu0kXJkL3_wb8foAba5VQWXVHhaPQ1V6Mh85oBciQwsPcHt1fhikqzWEUzIP8F2HCOJ9SWyki43vuWwUcSoAtFXhckncUHq0bTsiSU_Qz7W-Epc62h2KECf4mGErtDyk1oXMfeQH2UIP19F4gchagbrG7t99WH94GmoCNstfPg_MmH9CycHsBZHN9mXwrpaWKf1adjA0FzMT4i8SVSZ8bjkW29YHMZLA3dSypNhDN38NwjEqPYgsRSTKf5EWQhT9yBUJ5wgyrKYIiQSxzmSLCuUpVrcVZ7qgQZEiRIziVs7DOonCqZ0qaBZiIoDhkfeiyWFFw2sPQmoSJ-7fuKmK_8tywJjxhFU1fJKwX0e3amVPSqStjW45BmIw_nFi_vF3YSJOcfOnJFX2SJmZMx-5A9aIGCvMVy407F9qXTSuiMjmoyuPRxbaFPK-1TAlddUR0-o6LoUzhIHknupvcThrCnpHNMsCOaqH9FkmuruG-3hZyMhMIs9DsdoWMJcLsO-LBrsJPdQNa2xawxMd_z0GUh0z3v2cz5S9709k54Y4Cl9XGxOZMjQ1CZkgLJ4P_jUKUoVvl6vh-GEepVSHoF5Bflz37sCbAVqam4Th-Ycj3qjBoN98uJtaLLAuze67cgSw=s800-no" width=400>
           
           Tương tự bài toán phân loại 2 lớp, là đầu ra chúng ta cần một điểm thuộc 1 lớp nhất định và điều khác biệt ở đây đó là số lượng lớp luôn lớn hơn 2. Ở trong ví dụ này là 3 lớp. Các model phân biệt ảnh lớn có thể có đến 1000 lớp như [ImageNet](http://www.image-net.org/).

      - **Cách tiếp cận bài toán này**
      
         Tôi sẽ lấy bài phân loại 2 lớp để các bạn hình dung trước. Có 2 cách tiếp cận để giải bài toán này:

         | Discriminative model	 | Generative model|
         | ------ | ------ |
         | Tìm đường ngăn cách (Decision Boundary)	 | Tìm phân bố xác suất của dữ liệu (Probability distributions of the data)|  
         | <img src="https://lh3.googleusercontent.com/7DTs01OsbYlxFWjYgpva10MuVbgYBp3-wONSrDB-TM86w041bW4fAazswU4AQui4qKlDhtGQw0fFtFZGT6bPwzWrtAH85eYbszq9X0q-53vgV8AGbn5y3QJGar53_35GaZ9EuPU23oyxDCzFp2z3mQ_rd_bORqs4WHOtC9U3cqqE2I262ib7sy1XkBQoDI1vIry0Cs9SYA8Ka3YrOK6OTKo6ZYjbqlk0uCspkze9sx4o_7sc5dJjw90GMhQ_JFWFtVAZxm1qFuDUx7V3rFxly1d7QCvhZir_MwXfHy97rE-nKCGA_3wbbQnoM9WOIHkZER3UBH6h4xMMGqwHF4M96v1JibVh_01eBnttTYySfIymOJV8yK8m3qyiqhtH4WxHqx9tUjvgVxI8QEz9ET56dDDGkx_Sb24raSdOmB6IOuPj255n-u9CiJEI6qV9jPK9KXWiaLL1RQv0bVPvWpyTPexDNqUQQy1Y1xGKt3IKf0IJMft0yCM8Wo6oTGRNiLlglaakih2H7J7FxJvgHvq191gdTm_GMek0eSl8HuuZmzUpOXV_a7EkjFFqLdhdr2UyHdjnxobioc_t48l2QElq0NS3SXyW5nbLtpU4dTvpJ7pi_kthEsmc4m2fgBEbsA7V9bNmp7s54r5IcjtauTohNPUKzEvRqLysm39IR-DImKjINlI93CZ48MYTn9E2rlkuzlU6SS5rkAwwSxQhDw=w500-h300-no">	 | <img src="https://lh3.googleusercontent.com/cnzJHljtpTIop68rQiiMx3Y4Oel7g3tHDUmsdvvG-Ca_BWXp6a9Swo7aqdMskYgsX4puyTT0Kb3imvVzdpqLUQ_vyJ1EqJwRdSQwl-MU1QfNqoYxeTFhsluiucaOFsH4m3wfkQdkMqA1hXzrsoOxLkIWN5Nf88Z6l1cVghE-oSa8_olTY1qsq0rJPjjXHZNWHVO9a5QMHb5Uh7KPlayNpiAIMRtFDC1xRuYgctuFodndZGd-GJWyI-qwtJWZoCMkTM0eNJhDXlXJtXMMURwwSCP3vxHPhX2Vg-AqeG-_9BKgvQyQ8_bz342qVDcWyyOFtfoxF9nhHufCacuAyHgMBKAPs543bwQo6aDWSYEtWVPWA-oNSOjtJY7p7LIAFCoKTWmW3fVZjp-VMFyiQo2gIlLDMsZsC2htYezxQGb6Wk9PSmifZYHUzVQokZmi2kXk-i84hyVOoONxNusSZtBrMvZcC5oeC9_IL5aINkpw_oAb-jW1nw8drdNffCtAU614namIbFTzoiwe6TxkNoXtAfARKsCzKmsVuBTfJthRw92emSSZrUH9ehnkAbhjCUuFw_3Ozf91e_ghXSH26kASnh-GaKE172yS2TgwNZvxprJXgGicBzNEGLYhqrxyABNnpyRGLZ7ZB9Vea2dVRfnDLw6W0pmaJbLu_byVYl41t63PUMDosNrJIJ-J5DgABFNgGjZEpJQflFKGgUYG_g=w500-h300-no">|  

         Theo [Supervised Learning cheatsheet](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning#linear-models)
         
      - **Tiếp cận theo hướng Discriminative**
		Nhắc lại chúng ta sẽ tìm một đường ngăn cách để có thể phân biệt 2 vùng data với trường hợp phân loại 2 lớp hoặc nhiều vùng data trong trường hợp phân loại nhiều lớp.

		Trong trường hợp phân loại 2 lớp:
		
		<img src="https://lh3.googleusercontent.com/A67h2lsxXgkXLWeNDoC2b9IzL5iGFegRDtBPdulR8H1JVybn0dKXyJHW0DVvyw_AkuAjQ4gwhJibAcg4tKgoj2ABLB_bXl6WfDFWkP8-3UXTmAdUYvmc7ug6DwO37_VoeKhbgwHYxqh9NNuKQbsF0l_DSMC4HwmZ4bEJEjSkDqMhiDxkbB6a0AYr51UzB49UUYDgviIfptVu9tGJV8sKxF7RcP9ByO_C9N5fK6lBSWmgQz00w30IJVDFL_1Fow47JjSiV8midSh_ptpqB54lnF2tFSidOAwUkGsiFhQYeXZSWiHF1iFpw01O5jRbsnJNlpKlR6L8EG3gNHZITDSNkoZRph2iZj4fsP9uyEWpEvBSRD0GCckZZxFqMM7YkrVi6kZclJwK8YQI0pnMiAAQy2LtAVMKitw-qusHU9DCxyJgpuXU7asYd879LXb8vEWnBV2_ouvPibinkxSpJMAN3bHz-9lMcDysWdGovcOQuVP7iXeoHmcTC9F7BBxlL-s2zqNSKy7VdY8Vw-XBbkWYz70Nq3ZX7xrjmcaXPaMiNsJF4p0SFayzoGfI1zaoOhhEApqdc5zM6fkaTfuOmfAjUZQUN88aqQOEHLyC8CHyk-8odzOqbAfldUxTTap79XSJVzAFm-ghWJA7Hm_y_oO9uxUGy8dv9UMmqGCJ-EEF9-QkoKucNO9wvVSQJtsQmsUiGKW8B6ZdwwwYG2wWaw=s800-no" width=400>
	
		Phân bố xác suất:
		
		Xác suất 1 điểm là màu xanh:
	
		<img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{120}&space;\LARGE&space;P(y=1|x)&space;=&space;h_\theta(x)&space;=&space;g(\theta^Tx)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-\theta^{T}x}}&space;=&space;\frac{1}{1&space;&plus;&space;e^{-z}}" title="\LARGE P(y=1|x) = h_\theta(x) = g(\theta^Tx) = \frac{1}{1 + e^{-\theta^{T}x}} = \frac{1}{1 + e^{-z}}" />
	
		Trong đó: <img src="https://latex.codecogs.com/svg.latex?\inline&space;\LARGE&space;z&space;=&space;\theta^T{x}" title="\LARGE z = \theta^T{x}" />
		
		Xác suất 1 điểm là màu đỏ: 

		<img src="https://latex.codecogs.com/svg.latex?\inline&space;\LARGE&space;P(y=0|x)&space;=&space;1&space;-&space;P(y=1|x)&space;=&space;1&space;-&space;\frac{1}{1&space;&plus;&space;e^{-z}}" title="\LARGE P(y=0|x) = 1 - P(y=1|x) = 1 - \frac{1}{1 + e^{-z}}" />

