// Initialize map with India's coordinates
var map = L.map('map').setView([21, 78], 5); // Default zoom on India

// Add OpenStreetMap tiles
var osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
});

// Google Maps tiles
var googlestreets = L.tileLayer('http://{s}.google.com/vt?lyrs=m&x={x}&y={y}&z={z}', {
    maxZoom: 20,
    subdomains: ['mt0', 'mt1', 'mt2', 'mt3']
});

// Add Google Maps tiles to map
googlestreets.addTo(map);

// User marker & accuracy circle
var userMarker = null;
var userCircle = null;

// Function to update user location
function updateLocation(position) {
    var lat = position.coords.latitude;
    var lng = position.coords.longitude;
    var accuracy = position.coords.accuracy;

    console.log("User Location -> Latitude:", lat, ", Longitude:", lng, ", Accuracy:", accuracy);

    // If marker & circle exist, update their position
    if (userMarker) {
        userMarker.setLatLng([lat, lng]);
        userCircle.setLatLng([lat, lng]);
        userCircle.setRadius(accuracy);
    } else {
        // First time: create marker & circle
        userMarker = L.marker([lat, lng]).addTo(map);
        userCircle = L.circle([lat, lng], {
            color: '#3399ff',
            fillColor: 'lightblue',
            fillOpacity: 0.3,
            radius: accuracy
        }).addTo(map);
    }

    // Center map to user location
    map.setView([lat, lng], 17);

    // Send live location to Django backend
    fetch('/update_location/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        },
        body: JSON.stringify({ latitude: lat, longitude: lng })
    }).then(response => response.json())
      .then(data => console.log("Location Sent to Server:", data))
      .catch(error => console.error("Error:", error));
}

// Error handling for location access
function handleLocationError(error) {
    console.warn("Error fetching location:", error.message);
}

// Request live location tracking
if (navigator.geolocation) {
    navigator.geolocation.watchPosition(updateLocation, handleLocationError, {
        enableHighAccuracy: true
    });
} else {
    alert("Geolocation is not supported by your browser.");
}

// Function to get CSRF token for POST request
function getCSRFToken() {
    return document.cookie.split('; ')
        .find(row => row.startsWith('csrftoken='))
        ?.split('=')[1];
}
