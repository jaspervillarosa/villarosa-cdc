const supportsFileSystemAccessAPI = 
'getAsFileSystemHandle' in DataTransferItem.prototype;

const elem = document.querySelector(".container-Capture")
const fileCapture = document.querySelector(".fileCapture")
const headerCapture = elem.querySelector(".headerCapture")
const reset = elem.querySelector(".resetCapture")
const predict = elem.querySelector(".predictCapture")
let targetDiv = document.querySelector('.displayContainer')
let imgContainer = document.querySelector('.imgContainer')

// prevent navigation 
elem.addEventListener('dragover', (e) => {
    e.preventDefault();
})


reset.addEventListener('click', (e) => {
    e.preventDefault()
    elem.classList.remove("active");
    fileCapture.value="";
    headerCapture.textContent="CaptureMedia";
    imgContainer.style.display = "none";
    targetDiv.style.display="flex";

})

fileCapture.addEventListener('click', (e) => {
    // e.preventDefault()
    if(fileCapture.value!=null){
        // headerCapture.textContent = "Image Captured"
        elem.classList.add("active")
    }
    else{
        headerCapture.textContent="Capture Media";
    }
})


