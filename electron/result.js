
var img = document.getElementById("rimg");
var c = document.getElementById("cate");
var text = document.getElementById("analysis");

img.src = "static/seg_img.jpg";
c.innerHTML=displayD(data);
text.innerHTML="<p>"+data[3]+"</p>";

function displayD(d) {
	var s0 = "<p>" + d[1][0]+": "+ d[0][0] + "<br>";
	var s1 = d[1][1]+": "+ d[0][1] + "<br>";
	var s2 = d[1][2]+": "+ d[0][2] + "<br>";
	var s4 = "<br>size: "+ d[4] + "</p>"
	s = "<h2>"+d[2]+"</h2>"+s0+s1+s2+s4;
	return s;
}


var sendbutton = document.getElementById("send");

sendbutton.addEventListener("click", function(){
	let xhr = new XMLHttpRequest();
    xhr.open('GET', 'http://localhost:5000/detect');
    xhr.send();
	
})
