function generateQRCode() {
	var data = document.getElementById("data").value
	var busqueda = eel.dogCat(data);
	document.getElementById("animal").innerHTML = "Amarillo";
}

function setImage(base64) {
	document.getElementById("qr").src = base64
}

async function recogImage(){
	var data = await eel.btn_ResimyoluClick()();
		if (data) {
			console.log(data)
			eel.dogCat(data)(
				function(ret){
					var info = 'Gesto: ' + ret
					document.getElementById("infoGesto").innerHTML = info
					//console.log(ret)
				});
		}
}

async function getFolder() {
	var dosya_path = await eel.btn_ResimyoluClick()();
		if (dosya_path) {
			console.log(dosya_path);
			document.getElementById("data").value = dosya_path;
		}
}

