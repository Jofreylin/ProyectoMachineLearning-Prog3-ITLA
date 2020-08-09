
async function recogImageGesture(){
	
	var buscador = document.getElementById("buscadorImagenes")
	var boton = document.getElementById("analizarImagen")
	var spinner = document.getElementById("progressSpinnerG")
	buscador.disabled = true
	boton.disabled = true
	spinner.hidden = false
	
	
	var info = "Este es el (GESTO): 0%"
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
			spinner.hidden = true
			/*eel.getGesture(data)(
				function(ret){
					var info = data
					document.getElementById("infoGesto").innerHTML = info
					//console.log(ret)
				});*/
		}
}

async function recogImageFace(){
	var selectt = document.getElementById("seleccionOption")
	selectt.disabled = true
	var buscador = document.getElementById("buscadorImagenFace")
	var boton = document.getElementById("analizarImagenFace")
	var spinner = document.getElementById("progressSpinner")
	
	buscador.disabled = true
	boton.disabled = true
	spinner.hidden = false

	var info = "Rostros detectados: ()"
	document.getElementById("infoFace").innerHTML = info

	var typefile = document.getElementById("typeFileHidden").innerHTML
	typefile = 'data:image/'+typefile+';base64,'
	var b64data = document.getElementById("b64FileHidden").innerHTML
	b64data = b64data.replace(typefile,'')
	console.log(b64data)
	var data = await eel.detectFaces(b64data)();
		if (data) {
			console.log(data)
			info = data[1]
			document.getElementById("infoFace").innerHTML = info
			document.getElementById("imageRecognized").src = data[0]
			buscador.disabled = false
			boton.disabled = false
			spinner.hidden = true
			selectt.disabled = false
			/*eel.getGesture(data)(
				function(ret){
					var info = data
					document.getElementById("infoGesto").innerHTML = info
					//console.log(ret)
				});*/
		}
}

function previewImage(input,idInfo,stringInfo){
	
	var reader = new FileReader()
	reader.onload = function(){
		var output = document.getElementById('imagePreviewed')
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
		var info = stringInfo
		document.getElementById(idInfo).innerHTML = info
		
	}
	reader.readAsDataURL(input.files[0])
}

function seleccionar(element){

	document.getElementById('registrar').hidden = element.value == 1 ? false : true;
	document.getElementById('identificarImagen').hidden = element.value == 2 ? false : true;
	document.getElementById('identificarWebCam').hidden = element.value == 3 ? false : true;
}

async function verificarRostro(element){
	var selectt = document.getElementById("seleccionOption")
	selectt.disabled = true
	var texto = document.getElementById('inputVerificar')
	var spinner = document.getElementById("progressSpinnerR")
	var info = document.getElementById("infoFaceR")
	var adver = document.getElementById("textoAdvertencia")
	info.innerHTML = ""

	var btnUpload = document.getElementById("btnUpload")
	spinner.hidden = false
	if(texto.value.length <= 0){
		console.log('El campo no puede estar vacio')
		spinner.hidden = true
	}else{
		texto.disabled = true
		element.disabled = true
		
		var data = await eel.searchFace(texto.value)();
		if (data) {
			console.log(data)
			var info2 = data
			info.innerHTML = info2
			adver.hidden = false
			btnUpload.hidden = false
			spinner.hidden = true
			selectt.disabled = false
		}
	}
}

async function uploadFace(element){
	
	var texto = document.getElementById('inputVerificar')
	var spinner = document.getElementById("progressSpinnerR")
	spinner.hidden = false

	if(texto.value.length <= 0){
		console.log('El campo no puede estar vacio')
		spinner.hidden = true
	}else{
		var selectt = document.getElementById("seleccionOption")
		selectt.disabled = true
		var info = document.getElementById("infoFaceR")
		var btnRegister = document.getElementById("btnRegistrar")
		info.innerHTML = ""
		element.disabled = true
		btnRegister.disabled = true
		
		var data = await eel.setFaces(texto.value,100)();
		if (data) {
			console.log(data)
			var info2 = data
			info.innerHTML = info2
			btnUpload.hidden = false
			spinner.hidden = true
			btnRegister.disabled = false
			element.disabled = false
			selectt.disabled = false
		}
	}
	
}

async function registrarFace(element){
	var selectt = document.getElementById("seleccionOption")
	selectt.disabled = true
	var texto = document.getElementById('inputVerificar')
	var info = document.getElementById("infoFaceR")
	var spinner = document.getElementById("progressSpinnerR")
	var btnUpload = document.getElementById("btnUpload")
	var btnVerificar = document.getElementById("btnVerificar")

	info.innerHTML = ""
	element.disabled = true
	spinner.hidden = false
	btnUpload.disabled = true
	texto.disabled = true
	btnVerificar.disabled = true

	var data = await eel.trainFace()();
		if (data) {
			console.log(data)
			var info2 = data
			info.innerHTML = info2
			btnUpload.hidden = true
			btnUpload.disabled = false
			spinner.hidden = true
			element.disabled = false
			texto.disabled = false
			btnVerificar.disabled = false
			selectt.disabled = false
		}
}

async function recogImageFaceWebCam(element){
	var selectt = document.getElementById("seleccionOption")
	selectt.disabled = true
	var info = document.getElementById("infoFaceR")
	var spinner = document.getElementById("progressSpinnerW")
	element.disabled = true
	spinner.hidden = false
	var data = await eel.detectCameraFaces()();
		if (data) {
			console.log(data)
			var info2 = data
			//info.innerHTML = info2
			element.disabled = false
			spinner.hidden = true
			selectt.disabled = false
		}
}
