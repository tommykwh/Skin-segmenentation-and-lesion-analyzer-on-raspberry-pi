/*
const PiCamera = require('pi-camera');
const myCamera = new PiCamera({
  mode: 'photo',
  output: `./test.jpg`,
  width: 480,
  height: 480,
  nopreview: true,
}); 

var video = document.getElementById("cam");
video.srcObject = myCamera;
if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
      video.srcObject = myCamera;
    })
    .catch(function (err0r) {
      console.log("Something went wrong!");
    });
}
*/
