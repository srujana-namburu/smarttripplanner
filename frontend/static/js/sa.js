function getCSRFToken() {
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
        document.cookie.split(";").forEach(cookie => {
            let [name, value] = cookie.trim().split("=");
            if (name === "csrftoken") {
                cookieValue = decodeURIComponent(value);
            }
        });
    }
    return cookieValue;
}

document.getElementById("analyze-btn").addEventListener("click", function () {
    const locationInput = document.getElementById("location-input").value.trim();

    if (!locationInput) {
        alert("Please enter a location.");
        return;
    }

    fetch("/frontend/sentiment_analysis/", {
        method: "POST",
        headers: { 
            "Content-Type": "application/json",
            "X-CSRFToken": getCSRFToken()
        },
        body: JSON.stringify({ location: locationInput })
    })
    .then(response => response.json())
    .then(data => console.log("Response:", data))  // Debugging
    .catch(error => console.error("Error:", error));
});
