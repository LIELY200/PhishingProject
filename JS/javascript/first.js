// window
// document
let photos = ["images/child.jpg", "images/img2.jpg"];
let currentPhoto = 0;

function leftClick() {
	currentPhoto = (currentPhoto - 1) % photos.length;
	if(currentPhoto < 0) {
		currentPhoto *= -1;
	}

	showImg();
}

function rightClick() {
	currentPhoto = (currentPhoto + 1) % photos.length;
	showImg();
}

function showImg() {
	let image = document.getElementById("img1");
	image.src = photos[currentPhoto];
}