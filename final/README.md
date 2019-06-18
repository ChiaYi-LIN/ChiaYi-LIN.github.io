# 前端程式設計 期末專題 (國企四 林家毅 B04704016)

## 期末專題主題簡介

當一群朋友一起出遊，多少會遇到互相借錢、代為支付的情況，在次數少的情況下大家還可以靠著記憶將帳務理清，但是如果有太多筆複雜的支出，到最後都會很難算出彼此欠錢金額和還錢方式，因此，這個網站的出現就是為了設計一款讓帳務不再錯亂的拆帳軟體。

## 使用方式

1. 新增群組成員

    * 一開始進入網頁，會在畫面左方看到群組成員欄位`Group Members`，從下方的輸入框新增成員並按下`Add`就可以將成員加入列表。
    
    * 最多可以新增八位成員。
    
    * 每個成員右邊都會有一個移除按鈕，按下就可以把成員移出群組。
    
    * 在任何時候都可以新增、移除成員。

2. 新增帳目

    * 成功新增成員後，會在右方出現新增帳目的欄位，需要填入的資料包含消費日期`Date`、消費主旨`Title`、消費總金額`Price`，以及每位成員個別的實際支付金額`Payment`和消費金額`Consumption`，最後按下新增按鈕`Add Nem Bill`就可以將該筆支出加入消費歷史。
    
    * `Date`、`Title`和`Price`欄位不可留白。
    
    * `Price`、所有成員`Payment`總和、所有成員`Consumption`總和須彼此相等。

3. 查看消費歷史

   * 每筆新增的消費紀錄都會被儲存在右方的消費歷史欄位。
   
   * 按下某筆消費右下角的刪除按鈕可刪掉該紀錄。

4. 還錢！！！

   * 每次新增或刪除消費歷史時，最右方的畫面都會即時更新目前群組成員彼此的欠錢狀況，並給出還錢方式。

## JavaScript 程式碼說明

### Vue.js

* 群組成員的處理是使用Vue.js完成，當按下`Add`按鈕時會觸發`addNewParticipant()`，將新成員的`id`、`name`新增到物件中，並透過`each-participant`和`new-bill`兩個模板產出html在原本的網頁上。

```javascript
var main = new Vue({
    el: '#main',
    data: {
        newParticipant: '',
        participants: [],
        nextParticipantId: 1
    },
    methods: {
        addNewParticipant() {
            if ((this.newParticipant.trim() != '') & (this.participants.length < 8)) {
                this.participants.push({ id: this.nextParticipantId++, name: this.newParticipant })
            }
            this.newParticipant = ''
        }
    }
})

Vue.component('each-participant', {
    template: '\
      <li>\
        {{ name }}\
        <button class="fas fa-trash-alt delete-member" v-on:click="$emit(\'remove\')"></button>\
      </li>\
    ',
    props: ['name']
})

Vue.component('new-bill', {
    template: '\
    <tr class="row m-0">\
    <th scope="col" class="col-lg-4 member my-auto">{{ participant.name }}</th>\
    <th scope="col" class="col-lg-4 pl-0"><input type="number" class="payment form-control m-0 activity-info" v-model="participant.billPayment"></th>\
    <th scope="col" class="col-lg-4 pl-0"><input type="number" class="consume form-control m-0 activity-info" v-model="participant.billConsume"></th>\
    </tr>\
    ',
    props: ['participant']
})
```

### JQuery

* 一開始進入網頁時只會顯示第一個欄位，其他欄位會被隱藏，直到顯示條件被觸發。

```javascript
$('#activity').hide()
$('#all-history').hide()
$('#result').hide()

```

* `Input`物件在輸入完成時按下`Enter`會產生跟點擊按鈕一樣的效果

```javascript
 $('#add-one').keypress(function(e) {
    if (e.which == 13) {
        $('#add-new-member').click()
    }
})

$('#add-new-member').keypress(function(e) {
    if (e.which == 13) {
        $('#add-new-member').click()
    }
})

$('#add-new-bill').keypress(function(e) {
    if (e.which == 13) {
        $('#add-new-bill').click()
    }
})
```

* 至少有一個成員時顯示新增帳目欄位

```javascript
$('#add-new-member').on('click', function() {
    if ($('.each-member').length > 1) {
        $('#activity').show()
    }
})
```

* 新增帳目的條件設定、刪除帳目功能，並及時更新還錢方式。一旦觸發任何一種功能就會顯示歷史消費欄位和還錢欄位。

```javascript
$('#add-new-bill').on('click', function() {
    $date = $('#new-event-date')
    $title = $('#new-event-title')
    $price = $('#new-event-price')
    $member = $('.member')
    $payment = $('.payment')
    $consume = $('.consume')

    noDate = ($($date).val() == '')
    noTitle = ($($title).val().trim() == '')
    noPrice = ($($price).val() == 0)

    if (noDate | noTitle | noPrice) {
        return 0
    }

    totalPrice = Number($($price).val())
    totalPayment = 0
    totalConsume = 0
    $($payment).each(function() {
        totalPayment += Number($(this).val())
    })

    $($consume).each(function() {
        totalConsume += Number($(this).val())
    })

    if ((totalPrice != totalPayment) | (totalPrice != totalConsume) | (totalPayment != totalConsume)) {
        return 0
    }
    $('#all-history').show()
    $('#result').show()

    $pDate = $('<p>').text($($date).val()).attr('class', 'my-0 font-weight-bold')
    $pTitle = $('<p>').text($($title).val()).attr('class', 'my-0 font-weight-bold')
    $pPrice = $('<p>').text('$' + $($price).val()).attr('class', 'my-0 font-weight-bold')
    $head = $('<div>').append($pDate).append($pTitle).append($pPrice)
    $div = $('<div>').attr('class', 'w-100').append($head)

    $table = $('<table>')
    for (i = 0; i < $($member).length; i++) {
        $tdMember = $('<td>').text($($($member)[i]).text()).attr('class', 'history-border history-member')
        $tdPayment = $('<td>').text($($($payment)[i]).val()).attr('class', 'history-border history-payment')
        $tdConsume = $('<td>').text($($($consume)[i]).val()).attr('class', 'history-border history-consume')
        $tr = $('<tr>').append($tdMember).append($tdPayment).append($tdConsume)
        $body = $('<tbody>').append($tr)
        $($table).attr('class', 'history-table w-100 my-2').append($body)
    }

    $delete = $('<button>').attr('class', 'fas fa-trash-alt delete-member delete-history mb-2')

    $tableContain = $('<div>').attr('class', 'my-4 history-contain')
    $($tableContain).append($div).append($table).append($delete)
    $('#all-history').append($tableContain)

    $($delete).on('click', function() {
        $(this).parent('div').remove()
        calculateTotal()
    })

    calculateTotal()
})
```

* 還錢方式的計算流程並顯示在最右方的欄位

```javascript
var calculateTotal = function() {
    $('#result').empty()
    memberList = []
    totalPayment = []
    totalConsume = []
    $historyMember = $('.history-member')
    $historyPayment = $('.history-payment')
    $historyConsume = $('.history-consume')

    for (i = 0; i < $($historyMember).length; i++) {
        currentMemberName = $($($historyMember)[i]).text()
        currentMemberPayment = Number($($($historyPayment)[i]).text())
        currentMemberConsume = Number($($($historyConsume)[i]).text())
        if (memberList.includes(currentMemberName) == false) {
            memberList.push(currentMemberName)
            totalPayment.push(currentMemberPayment)
            totalConsume.push(currentMemberConsume)
        } else {
            index = memberList.indexOf(currentMemberName)
            totalPayment[index] += currentMemberPayment
            totalConsume[index] += currentMemberConsume
        }
    }

    console.log(memberList)
    console.log(totalPayment)
    console.log(totalConsume)

    moneyToGet = []
    for (i = 0; i < memberList.length; i++) {
        moneyToGet.push(totalPayment[i] - totalConsume[i])
    }

    console.log(moneyToGet)

    result = []
    while (moneyToGet.some(item => item !== 0)) {
        max = Math.max(...moneyToGet)
        min = Math.min(...moneyToGet)

        fromIndex = moneyToGet.indexOf(min)
        toIndex = moneyToGet.indexOf(max)

        if (max >= Math.abs(min)) {
            moneyToGet[fromIndex] -= min
            moneyToGet[toIndex] += min
            result.push([memberList[fromIndex], memberList[toIndex], Math.abs(min)])
        } else {
            moneyToGet[fromIndex] += max
            moneyToGet[toIndex] -= max
            result.push([memberList[fromIndex], memberList[toIndex], Math.abs(max)])
        }
    }

    console.log(result)

    for (i = 0; i < result.length; i++) {
        $cashFlow = $('<div>').attr('class', 'text-center')
        $from = $('<p>').text(result[i][0]).attr('class', 'd-inline-block mr-3')
        $arrow = $('<i>').attr('class', 'fas fa-angle-double-right d-inline-block mr-3')
        $to = $('<p>').text(result[i][1]).attr('class', 'd-inline-block mr-3')
        $cash = $('<p>').text('$' + result[i][2]).attr('class', 'd-inline-block')
        $cashFlow.append($from).append($arrow).append($to).append($cash)
        $('#result').append($cashFlow)
    }
}
```
