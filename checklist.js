new Checklists("article", function(checkbox, callback) {
    var uri = checkbox.closest("article").find("h1 a").attr("href");
    jQuery.get(uri, callback);
}, function(markdown, checkbox, callback) {
    var uri = checkbox.closest("article").find("h1 a").attr("href");
    jQuery.ajax({
        type: "put",
        uri: uri,
        data: markdown,
        success: callback
    });
});