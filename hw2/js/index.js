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

    $('#all-products').on('click', function() {
        $('#product-list').empty()
        showItems(1, items)
        newPage(items.length, pageCount, items)
        $('#page').show()
    })

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

    $('#search-button').on('click', function() {
        if ($('#search').val().trim() == '') {
            var searchKeyWord = $('#search').attr('placeholder')
        } else {
            searchKeyWord = $('#search').val()
        }

        var searchItems = items.filter(function(item, index, array) {
            return item.name.toLowerCase().includes(searchKeyWord.toLowerCase())
        })

        if (searchItems.length != 0) {
            $('#product-list').empty()
            $('#add-product-form').hide()
            showItems(1, searchItems)
            newPage(searchItems.length, pageCount, searchItems)
            console.log(searchItems.length)
        }

    })

    $("#search").on('keypress', function(event) {
        if (event.keyCode == 13) {
            $("#search-button").click()
            return false
        }
    })

    var showItems = (page, items) => {
        if (items == null) return
        var start = (page - 1) * pageCount
        var end = start + pageCount - 1
        $('#product-list').empty()
        for (var i = start; i <= end; i++) {
            newItem(items[i])
        }
    }

    var newItem = (item) => {
        try {
            $img = $('<img>').attr('class', 'image').attr('src', item.image)
            $h3 = $('<h3>').attr('class', 'name').text(item.name)
            $p = $('<p>').attr('class', 'price').text('NT$ ' + item.price)

            $item = $('<div>').attr('class', 'item').append($img).append($h3).append($p)
            $col = $('<div>').attr('class', 'col').append($item)

            $('#product-list').append($col)
        } catch (e) {
            console.log('Error in loading items')
        }
    }

    var newPage = (n, pageCount, pageItems) => {
        var pageNum = n / pageCount
        pageNum = (n % pageCount != 0) ? pageNum + 1 : pageNum
        pageNum = Math.floor(pageNum)
        $('#page-number').empty()

        $la = $('<a>').attr('class', 'page-link').attr('href', '#').attr('tabindex', '-1').attr('aria-disabled', 'true').text('«')
        $lli = $('<li>').attr('class', 'page-item').addClass('disabled').append($la)
        $lli.on('click', function() {
            currentPage = Number($('li.page-item.active').text())
            if (currentPage > 1) {
                switchPage(currentPage, currentPage - 1, pageNum)
                currentPage = currentPage - 1
                showItems(currentPage, pageItems)
            }
        })

        $('#page-number').append($lli)

        for (var i = 1; i <= pageNum; i++) {
            $a = $('<a>').attr('class', 'page-link').attr('href', '#').text(i)
            $a.on('click', function() {
                var i = $(this).text()
                showItems(Number(i), pageItems)
                switchPage(currentPage, i, pageNum)
                currentPage = i
            })

            var strActive = ((i == 1) ? ' active' : '')
            $li = $('<li>').attr('class', 'page-item' + strActive).append($a)
            $('#page-number').append($li)
        }

        $ra = $('<a>').attr('class', 'page-link').attr('href', '#').text('»')
        if (pageNum == 1) {
            $rli = $('<li>').attr('class', 'page-item disabled').append($ra)
        } else {
            $rli = $('<li>').attr('class', 'page-item').append($ra)
        }
        $rli.on('click', function() {
            currentPage = Number($('li.page-item.active').text())
            if (currentPage < pageNum) {
                switchPage(currentPage, currentPage + 1, pageNum)
                currentPage = currentPage + 1
                showItems(currentPage, pageItems)
            }
        })
        $('#page-number').append($rli)
    }

    var switchPage = function(fromPage, toPage, allPage) {
        $ali = $('li.page-item')
        $ali.removeClass()
        if (toPage == 1) {
            $lli.removeClass()
            $lli.addClass('page-item disabled')
        } else {
            $lli.removeClass()
            $lli.addClass('page-item')
        }

        if (toPage == allPage) {
            $rli.removeClass()
            $rli.addClass('page-item disabled')
        } else {
            $rli.removeClass()
            $rli.addClass('page-item')
        }

        $ali.addClass('page-item')
        $($ali[Number(toPage)]).addClass('active')

    }

    $('#add-product-submit').on('click', function() {
        if ($('#product-name').val().trim() == '' || $('#product-price').val().trim() == '' || $('#product-count').val().trim() == '' || $('#product-url').val().trim() == '') {
            $('#message').text('Please fill out all blanks and retry again.')
            $('#dialog').modal('show')
            return 0
        }

        var data = {
            item: {
                name: $('#product-name').val(),
                price: Number($('#product-price').val()),
                count: Number($('#product-count').val()),
                image: $('#product-url').val(),
            }
        }

        $.post('https://js.kchen.club/B04704016/insert', data, function(response) {
            if (response) {
                if (response.result) {
                    $('#message').text('Success')
                    console.log(response.item)
                    $('#dialog').modal('show')
                } else {
                    $('#message').text('Fail')
                    console.log(response.message)
                    $('#dialog').modal('show')
                }
            } else {
                $('#message').text('Server Error')
                $('#dialog').modal('show')
            }
        })
    })

})