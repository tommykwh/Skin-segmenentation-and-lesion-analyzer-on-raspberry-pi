// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// All of the Node.js APIs are available in this process.

var video = document.getElementById("cam");
var box = document.getElementById("box");
var loader = document.getElementById("loader");
var p = document.getElementById("prompt");

if (navigator.mediaDevices.getUserMedia && video) {
  navigator.mediaDevices.getUserMedia({ audio: false, video: true }) //{ width: 640, height: 480 }
    .then(function (stream) {
      
      box.style.visibility = "visible";
      video.srcObject = stream;
      
      /*
      var panel = document.createElement('div');
      panel.id = 'block';
      panel.className = 'right';
      document.getElementsByTagName('body')[0].appendChild(panel);
      */
      var take = document.getElementById("hitbox");
      var bk = document.getElementById("resultback");
      bk.addEventListener('click',function() {
        console.log("sssssshsit")
        window.location.replace("./JASMINE.html");
      });
      p.addEventListener('DOMSubtreeModified', openResult);
      
      take.addEventListener("click", function(){
        loader.classList.toggle("hide");
        console.log("shit");
        var draw = document.createElement("canvas");
        draw.width = video.videoWidth;
        draw.height = video.videoHeight;
        var context2D = draw.getContext("2d");
        context2D.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        var image = draw.toDataURL("image/png").replace("image/png", "image/octet-stream"); 
        //window.location.href=image; 
      });
    })
    .catch(function (err0r) {
      console.log("Something went wrong!");
    });
}



function openResult(){
   window.location.replace("RESULT.html");
}
