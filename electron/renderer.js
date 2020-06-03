// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// All of the Node.js APIs are available in this process.

var video = document.getElementById("cam");
var box = document.getElementById("box");
var loader = document.getElementById("loader");
var pl = document.getElementById("pl");
var prompt = document.getElementById("prompt");
//var p = document.getElementById("prompt");

pl.classList.toggle("hide");
function postFile(file) {
    let formdata = new FormData();
    formdata.append("image", file);
    let xhr = new XMLHttpRequest();
    xhr.open('POST', 'http://localhost:5000/detect', true);
    xhr.onload = function () {
        if (this.status === 200) {
            console.log(this.response);
            result = localStorage.getItem(this.response);
            openResult();
        }
        else
            console.error(xhr);
    };
    xhr.send(formdata);
}


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
      
      //p.addEventListener('DOMSubtreeModified', openResult);
      
      take.addEventListener("click", function(){
        
        //raspistill
        loader.classList.toggle("hide");
        pl.classList.toggle("hide");
        prompt.innerHTML = "Analyzing..."
        var draw = document.createElement("canvas");
        draw.width = video.videoWidth;
        draw.height = video.videoHeight;
        var context2D = draw.getContext("2d");
        context2D.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        draw.toBlob(postFile, 'image/jpg');
        //var image = draw.toDataURL("image/png").replace("image/png", "image/octet-stream"); 
        //window.location.href=image; 
        //link.href = image;
        //link.download = "capture.png";
      });
    })
    .catch(function (err0r) {
      console.log("Something went wrong!");
    });
}


function openResult(){
   window.location.replace("RESULT.html");
}


