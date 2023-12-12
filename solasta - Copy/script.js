
document.addEventListener("DOMContentLoaded", function () {
    var preloaderDuration = 3250;
    document.getElementById("preloader").style.display = "flex";
    setTimeout(function () {
        document.getElementById("preloader").style.display = "none";
    }, preloaderDuration);
});




