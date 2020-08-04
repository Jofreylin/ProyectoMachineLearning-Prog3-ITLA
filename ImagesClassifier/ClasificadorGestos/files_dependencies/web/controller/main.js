function getGesture() {
	var data = document.getElementById("data").value
	var busqueda = eel.getGesture(data);
	document.getElementById("animal").innerHTML = "Amarillo";
}

function setImage(base64) {
	document.getElementById("qr").src = base64
}

async function recogImage(){
	var buscador = document.getElementById("buscadorImagenes")
	var boton = document.getElementById("analizarImagen")
	var barra = document.getElementById("progressBarGestos")
	
	buscador.disabled = true
	boton.disabled = true
	barra.hidden = false;
	barra.innerHTML="0%"
	barra.style.width = "0%"

	var info = "(Este es el GESTO): 0%"
	document.getElementById("infoGesto").innerHTML = info

	var typefile = document.getElementById("typeFileHidden").innerHTML
	typefile = 'data:image/'+typefile+';base64,'
	var b64data = document.getElementById("b64FileHidden").innerHTML
	b64data = b64data.replace(typefile,'')
	console.log(b64data)
	var data = await eel.getGesture(b64data)();
		if (data) {
			console.log(data)
			info = data
			document.getElementById("infoGesto").innerHTML = info
			buscador.disabled = false
			boton.disabled = false
			barra.style.width = "100%"
			barra.innerHTML="100%"
			/*eel.getGesture(data)(
				function(ret){
					var info = data
					document.getElementById("infoGesto").innerHTML = info
					//console.log(ret)
				});*/
		}
}

async function getFolder() {
	var dosya_path = await eel.btn_ResimyoluClick()();
		if (dosya_path) {
			console.log(dosya_path);
			document.getElementById("data").value = dosya_path;
		}
}

function previewImage(input){
	
	var reader = new FileReader()
	reader.onload = function(){
		var output = document.getElementById('imageGesto')
		output.src = reader.result
		var b64 = output.src
		var typefile = input.files[0].type.split('/')[1] 
		//var typefile2 = document.getElementById('inputGroupFile01').value.split('.').pop().toLowerCase()
		console.log(typefile)
		console.log(b64)
		document.getElementById("typeFileHidden").innerHTML = typefile
		document.getElementById("b64FileHidden").innerHTML = b64
		typefile = 'data:image/'+typefile+';base64,'
		b64 = b64.replace(typefile,'')
		var info = "(Este es el GESTO): 0%"
		document.getElementById("infoGesto").innerHTML = info
		
	}
	reader.readAsDataURL(input.files[0])
}