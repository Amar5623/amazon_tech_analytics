// frontend_final\js\api.js


// Base URLs for different API endpoints
const ANALYTICS_BASE = 'http://localhost:8000/api/analytics';
const BRAND_BASE = 'http://localhost:8000/api/brand';
const ML_BASE = 'http://localhost:8000/ml';

// API utility object
export const api = {
    // GET request with query parameters
    async get(endpoint, params = {}) {
        try {
            // Build query string from params
            const queryString = Object.keys(params)
                .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`)
                .join('&');
            
            const url = `http://localhost:8000${endpoint}${queryString ? '?' + queryString : ''}`;
            
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`Error fetching ${endpoint}:`, error);
            throw error;
        }
    },
    
    // POST request
    async post(endpoint, data = {}) {
        try {
            // The endpoint might already include query parameters
            const url = `http://localhost:8000${endpoint}`;
                
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
                
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`API error: ${response.status} ${response.statusText} - ${errorText}`);
            }
                
            return await response.json();
        } catch (error) {
            console.error(`Error posting to ${endpoint}:`, error);
            throw error;
        }
    }
};

// Analytics API functions
export async function getPriceVsRating() {
    const res = await fetch(`${ANALYTICS_BASE}/price-vs-rating`);
    return await res.json();
}

export async function getTopBrands() {
    const res = await fetch(`${ANALYTICS_BASE}/top-brands`);
    return await res.json();
}

export async function getDiscountVsRating() {
    const res = await fetch(`${ANALYTICS_BASE}/discount-vs-rating`);
    return await res.json();
}

export async function getReviewDistribution() {
    const res = await fetch(`${ANALYTICS_BASE}/review-distribution`);
    return await res.json();
}

// ML API functions
export const mlAPI = {
    // Get available ML algorithms
    async getAvailableAlgorithms() {
        return await api.get(`/ml/available-algorithms`);
    },
    
    // Get data statistics for ML
    async getDataStats() {
        return await api.get(`/ml/data-stats`);
    },
    
    // Check if an algorithm is compatible with the dataset
    async checkModelCompatibility(algorithm, targetColumn = null) {
        const params = { algorithm };
        if (targetColumn) {
            params.target_column = targetColumn;
        }
        return await api.get(`/ml/model-compatibility`, params);
    },
    
    // Run supervised learning algorithm
    async runSupervisedLearning(data) {
        return await api.post(`/ml/supervised-learning`, data);
    },
    
    // Run clustering algorithm
    async runClustering(data) {
        return await api.post(`${ML_BASE}/clustering`, data);
    },
    
    // Run association rules mining
    async runAssociationRules(data) {
        return await api.post(`/ml/association-rules`, data);
    },
    
    // Get feature importance
    async getFeatureImportance(targetColumn) {
        // Create the URL with query parameter
        const url = `/ml/feature-importance?target_column=${encodeURIComponent(targetColumn)}`;
        // Make a POST request with an empty body but with the query parameter in the URL
        return await api.post(url, {});
    }
};

// Helper functions for ML visualizations
export function showLoading(elementId, message = 'Loading...') {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<div class="loading">${message}</div>`;
    }
}

export function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<div class="error-message">${message}</div>`;
    }
}

export function displayImage(containerId, base64String, altText = 'Visualization') {
    const container = document.getElementById(containerId);
    if (container && base64String) {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${base64String}`;
        img.alt = altText;
        img.className = 'visualization-image';
        container.appendChild(img);
    }
}

// This is a partial implementation focusing on the displayImage function
// Add this to your existing api.js file if it's not already there

export function displayImage2(containerId, base64String, altText = 'Visualization') {
    const container = document.getElementById(containerId);
    
    if (!container) {
        console.error(`Container with ID '${containerId}' not found`);
        return;
    }
    
    if (!base64String) {
        console.error('No base64 image data provided');
        return;
    }
    
    // Create image element
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${base64String}`;
    img.alt = altText;
    img.className = 'visualization-image';
    
    // Add to container
    container.appendChild(img);
    
    // Log success
    console.log(`Image added to container '${containerId}'`);
}
