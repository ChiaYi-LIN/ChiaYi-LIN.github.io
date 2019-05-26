# 前端程式設計 作業二

## 查詢
* 查詢所有商品
  * 開啟網頁時自動從資料庫載入商品，並顯示第一頁
    `$('#all-products').on('click', function() {
        $('#product-list').empty()
        showItems(1, items)
        newPage(items.length, pageCount, items)
        $('#page').show()
    })`
  * 按下 Search Products 按鈕會把已經下載過的產品資訊再一次輸出到頁面
    
  
* 分類查詢
* 自訂查詢

## 分頁

## 新增
