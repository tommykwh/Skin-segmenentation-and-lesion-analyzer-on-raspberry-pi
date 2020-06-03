
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
        loader.classList.toggle("hide");
        var draw = document.createElement("canvas");
        draw.width = video.videoWidth;
        draw.height = video.videoHeight;
        var context2D = draw.getContext("2d");
        context2D.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        var image = draw.toDataURL("image/png").replace("image/png", "image/octet-stream"); 
        //window.location.href=image; 
        //link.href = image;
        //link.download = "capture.png";
        myCamera.snap();
      });
    })
    .catch(function (err0r) {
      console.log("Something went wrong!");
    });
}



function openResult(){
   window.location.replace("RESULT.html");
}
