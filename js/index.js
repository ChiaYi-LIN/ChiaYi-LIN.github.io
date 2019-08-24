$('#send-email').on('click', function() {
    var sender_name = $('#sender-name').val()
    var sender_url = $('#sender-url').val()
    var sender_subject = $('#sender-subject').val()
    var sender_body = $('#sender-body').val()
    var footer

    if (sender_name == '' && sender_url == '') {
        footer = ''
    } else if (sender_name == '') {
        footer = 'From ' + sender_url
    } else if (sender_url == '') {
        footer = 'By ' + sender_name
    } else {
        footer = 'By ' + sender_name + ' from ' + sender_url
    }

    window.location.href = 'mailto:chiayi.lin.jack@gmail.com?subject=' + sender_subject + '&body=' + sender_body + '%0D%0A' + '%0D%0A' + footer
})