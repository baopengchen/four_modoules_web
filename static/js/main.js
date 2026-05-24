const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const thresholdRange = document.getElementById("thresholdRange");
const thresholdValue = document.getElementById("thresholdValue");
const predictForm = document.getElementById("predictForm");
const resetBtn = document.getElementById("resetBtn");
const compareBox = document.getElementById("compareBox");
const compareRange = document.getElementById("compareRange");
const compareOverlay = document.getElementById("compareOverlay");
const compareHandle = document.getElementById("compareHandle");

if (fileInput && fileName) {
    fileInput.addEventListener("change", () => {
        fileName.textContent = fileInput.files.length > 0
            ? fileInput.files[0].name
            : "选择白细胞显微图像";
    });
}

if (thresholdRange && thresholdValue) {
    thresholdValue.textContent = Number(thresholdRange.value).toFixed(2);
    thresholdRange.addEventListener("input", () => {
        thresholdValue.textContent = Number(thresholdRange.value).toFixed(2);
    });
}

if (predictForm && fileInput) {
    predictForm.addEventListener("submit", (event) => {
        if (!fileInput.files || fileInput.files.length === 0) {
            event.preventDefault();
            alert("请先选择一张 jpg、jpeg 或 png 格式的白细胞显微图像。");
        }
    });
}

if (resetBtn) {
    resetBtn.addEventListener("click", () => {
        window.location.href = "/";
    });
}

function updateCompare(value) {
    if (!compareOverlay || !compareHandle) {
        return;
    }
    compareOverlay.style.width = `${value}%`;
    compareHandle.style.left = `${value}%`;
}

function syncCompareImageWidth() {
    if (!compareBox || !compareOverlay) {
        return;
    }
    const overlayImg = compareOverlay.querySelector("img");
    if (overlayImg) {
        overlayImg.style.width = `${compareBox.clientWidth}px`;
    }
}

if (compareRange) {
    syncCompareImageWidth();
    updateCompare(compareRange.value);
    compareRange.addEventListener("input", () => {
        updateCompare(compareRange.value);
    });
    window.addEventListener("resize", syncCompareImageWidth);
}
