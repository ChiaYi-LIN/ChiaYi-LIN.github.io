# 前端程式設計 作業二

## 功能簡介
### 查詢
* 查詢所有商品
  * 開啟網頁時自動從資料庫載入商品，並顯示第一頁
  * 按下 `Search Products` 會把已經下載過的產品資訊再一次輸出到頁面
  
       使用分類查詢、自訂查詢、換頁或新增功能後，如果想要回到一開始包含所有商品的頁面，可以按下 `Search Products` 來完成
   
  * 按下畫面左上角的 Logo 會刷新頁面
      
       使用新增功能後，可以按下 Logo 刷新頁面來顯示新加入的新商品
    
* 分類查詢
    * 點擊 `Search By Category` 可以開啟下拉式表單，並出現預設分類
    
    * `Mobile`, `Home Automation`, `Power`, `Headphone`, `Accessories` 為預設分類，點擊後會顯示各類別商品
    
    * 點擊 `All Products` 會顯示頁面載入成功時從資料庫取得的所有商品，功能與 `Search Products` 相同
    
* 自訂查詢
    * 在畫面右上角的白色方塊中可以輸入自訂的商品搜尋關鍵字，輸入後按下 `Enter` 或是點擊旁邊的 `Search` 按鈕即可進行搜尋

### 分頁
* 指定頁數
    * 點擊每一頁底下的數字，都可以跳轉至該頁
    
    * 當下的頁數會以籃底呈現，其他可以前往的頁面則以白底呈現

* 上一頁、下一頁
    * 點擊分頁列最左邊的 `<<` 按鈕，可以前往上一頁，或是點擊最右邊的 `>>` 按鈕來前往下一頁
    
    * 當頁面在第一頁的時候 `<<` 的功能會被取消，在最後一頁時 `>>`的功能會被取消


### 新增
* 新增功能
    * 點擊 `Add Products` 按鈕可以進入新增功能

    * 根據 API ，在輸入 `商品名稱`、`商品價格`、`商品數項` 和 `商品圖片網址` 後，按下 `Add to server` 送出 post request
    
* 新增條件
    * 如果有任何一個欄位為空白，則按下送出按鈕時系統會彈出要求使用者確認輸入值的視窗

### 功能切換
* 查詢和新增功能之間的切換
    * 網頁載入時預設顯示查詢頁面，在導覽列上有 `查詢 (Search Products)` 和 `新增 (Add Products)` 兩種模式可以選擇
    
    * 在查詢模式中，按下 `Add Products` 會進入新增模式，此時導覽列的 `Search By Category` 和 `Search` 會被隱藏，同時頁面會顯示新增表單
    
    * 在新增模式中，按下 `Search Products` 會進入查詢模式，此時導覽列的 `Search By Category` 和 `Search` 會重新顯示，同時頁面會顯示一開始網頁載入的所有商品

## JavaScript 程式碼說明
* 載入頁面完成後，隱藏新增和分頁功能，進行查詢並顯示產品
```javascript
$(document).ready(function() {
    $('#add-product-form').hide()
    $('#page').hide()
    var items = null
    var pageCount = 20
    var currentPage = 1

    $.get('https://js.kchen.club/B04704016/query', function(response) {
        if (response) {
            if (response.result) {
                items = response.items
                $('#product-list').empty()
                showItems(1, items)
                newPage(items.length, pageCount, items)
                $('#page').show()

            } else {
                $('#message').text('查無相關資料')
                $('#dialog').modal('show')
            }
        } else {
            $('#message').text('伺服器出錯')
            $('#dialog').modal('show')
        }

        console.log(response)
    }, "json")
})
```

* 建立分類查詢的關鍵字字典
```javascript
$('#mobile').on('click', function() {
    categoryFilter(items, ['紅米', '小米4', '小米5', '小米平板', '小米6', '小米Max', '小米MIX', '小米Note'])
})

$('#home-automation').on('click', function() {
    categoryFilter(items, ['路由', '空氣', '檯燈', '盒子', '鬧鐘', '體重', '淨水', '電視', '音箱', '枕', '巾', '床'])
})

$('#power').on('click', function() {
    categoryFilter(items, ['電源', '充電', '傳輸線'])
})

$('#headphone').on('click', function() {
    categoryFilter(items, ['耳機'])
})

$('#accessories').on('click', function() {
    categoryFilter(items, ['手環', '包', '隨身', '太陽鏡', '短袖', '口罩'])
})
```

* 分類查詢功能函數，會搜尋產品名稱至少包含一個字典內容的產品
```
categoryFilter = function(fromItems, key) {
    all_results = []
    for (i = 0; i < key.length; i++) {
        result = fromItems.filter(function(item, index, array) {
            return item.name.toLowerCase().includes(key[i])
        })
        all_results.push.apply(all_results, result)
    }
    $('#product-list').empty();
    showItems(1, all_results)
    newPage(all_results.length, pageCount, all_results)
    $('#page').show()
}
```

* 查詢、新增功能切換
```javescript
$('#search-function').on('click', function() {
    $('#search-dropdown').show()
    $('#search').show()
    $('#search-button').show()
    if ($('#add-function').hasClass('active')) {
        $('#add-function').removeClass('active')
    }
    if ($('#search-function').hasClass('active') == false) {
        $('#search-function').attr('class', ' active')
    }
    $('#product-list').empty()
    $('#add-product-form').hide()
    showItems(1, items)
    newPage(items.length, pageCount, items)
    $('#page').show()
})

$('#add-function').on('click', function() {
    $('#search-dropdown').hide()
    $('#search').hide()
    $('#search-button').hide()
    if ($('#search-function').hasClass('active')) {
        $('#search-function').removeClass('active')
    }
    if ($('#add-function').hasClass('active') == false) {
        $('#add-function').attr('class', ' active')
    }
    $('#product-list').empty()
    $('#add-product-form').show()
    $('#page').hide()
})
```
