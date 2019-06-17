Vue.component('history', {
    template: '\
    <div>\
    {{bill.date}} {{bill.title}} {{bill.price}} {{bill.participant}} {{bill.payment}} {{bill.consume}}\
    <button class="fas fa-trash-alt" v-on:click="$emit(\'remove\')"></button>\
    </div>\
    ',
    props: ['bill']
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

var main = new Vue({
    el: '#main',
    data: {
        newParticipant: '',
        participants: [],
        nextParticipantId: 1,
        bills: [],
        billDate: '',
        billTitle: '',
        billPrice: 0,
        billParticipant: [],
        billPayment: 0,
        billConsume: 0
    },
    methods: {
        addNewParticipant() {
            if ((this.newParticipant.trim() != '') & (this.participants.length < 8)) {
                this.participants.push({ id: this.nextParticipantId++, name: this.newParticipant })
            }
            this.newParticipant = ''
        },
        addNewBill() {
            //     // if (this.billDate == '' | this.billTitle.trim() == '' | this.billPrice == 0) {
            //     //     return
            //     // }
            //     let payment = []
            //     let consume = []
            //     for (i = 0; i < this.participants.length; i++) {
            //         this.billParticipant.push(this.participants[i].name)
            //     }

            //     // payment.push(this.billPayment)
            //     // consume.push(this.billConsume)
            //     this.bills.push({
            //         date: this.billDate,
            //         title: this.billTitle,
            //         price: this.billPrice,
            //         participant: this.billParticipant,
            //         payment: payment,
            //         consume: consume
            //     })
            //     this.billDate = ''
            //     this.billTitle = ''
            //     this.billPrice = 0
            //     this.billParticipant = []
        }
    }
})

$(function() {
    $('#activity').hide()
    $('#all-history').hide()
    $('#result').hide()

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

    $('#add-new-member').on('click', function() {
        if ($('.each-member').length > 1) {
            $('#activity').show()
        }
    })

    $('#add-new-bill').keypress(function(e) {
        if (e.which == 13) {
            $('#add-new-member').click()
        }
    })

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
})