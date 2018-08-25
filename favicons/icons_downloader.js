let iconElements = [...document.querySelectorAll("[id|=icon]")];
let icon_url = "https://www.iconfinder.com/icons/{icon-id}/download/png/128";
let urls = [];

iconElements.forEach(function(iconDiv) {
	let iconId = iconDiv.id.replace("icon-", "");
	urls.push(icon_url.replace("{icon-id}", iconId))
});

document.addEventListener("copy", function(e) {
	// Using the copy event, since it is the only time when we can push something
	// to the clipboard
	e.clipboardData.setData("text/plain", urls.join("\n"));
	e.preventDefault();
	console.log(urls.length, " copied successfuly");
});