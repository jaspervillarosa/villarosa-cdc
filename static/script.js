const supportsFileSystemAccessAPI = 
'getAsFileSystemHandle' in DataTransferItem.prototype;

const elem = document.querySelector(".container-dragDrop")
const fileInput = document.querySelector(".fileInput")
const dragText = elem.querySelector(".header")
const reset = elem.querySelector(".reset")
const predict = elem.querySelector(".predict")
let imgDisplay = document.querySelector('.img')
let forForm = document.querySelector(".forForm")
let graph = document.querySelector('.graph')
let result = document.querySelector(".result")

// prevent navigation 
elem.addEventListener('dragover', (e) => {
    e.preventDefault();
})

// Visually highlight the dropzone 
elem.addEventListener('dragenter', (e) => {
    e.preventDefault();
    elem.classList.add("active")
    dragText.textContent = "Release to Upload"

})

elem.addEventListener('dragleave', (e) => {
    elem.classList.remove("active")
    dragText.textContent = "Drag & Drop to Upload File"
})

elem.addEventListener('drop', (e) => {
    e.preventDefault()
    elem.classList.add("active")
    dragText.textContent = "File Uploaded"
    fileInput.files = e.dataTransfer.files
    console.log(fileInput.files)
    
})

reset.addEventListener('click', (e) => {
    e.preventDefault()
    elem.classList.remove("active");
    fileInput.value="";
    dragText.textContent="Drag & Drop to Upload File";
    imgDisplay.style.display = "none"
    graph.value = ''

})

predict.addEventListener('click', (e) => {
    // e.preventDefault()
    // result.style.display = 'flex'
    // // form.action = "predictImage/"
    // forForm.style.border = "solid blue" 
    // if (result.style.display !== "none"){
    //     forForm.style.display = "none";
    //     result.style.display = "block";
    // }else{
    //     forForm.style.display="none";
    //     result.style.display = "none";
    // }
})

fileInput.addEventListener('click', (e) => {
    if(fileInput.value!=null){
        dragText.textContent = "File Uploaded"
        elem.classList.add("active")
    }
    else{
        dragText.textContent="Drag & Drop to Upload File";
    }
})


