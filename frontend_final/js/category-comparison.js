// Get selected category
const categoryDropdown = document.getElementById("category-filter");
let selectedCategory = categoryDropdown.value;

// Map of category to spec_name for spec frequency endpoint
const categoryToSpecMap = {
    "laptops": "maximum memory supported",
    "smartphones": "ram",
    "smartwatches": "display_type",
    "wireless_earbuds": "battery_life"
};

// Event listener for category dropdown
categoryDropdown.addEventListener('change', (event) => {
    selectedCategory = event.target.value;
    console.log("Category changed to:", selectedCategory);
    updateAllCharts(selectedCategory);
});

// Main function to update all charts
function updateAllCharts(category) {
    console.log("Updating charts for category:", category);
    
    // Fetch category comparison data (price, rating, discount)
    getCategoryComparison()
        .then(data => {
            console.log("Category comparison data received:", data);
            renderMetricChart("avg-price-chart", data, "avg_price", "Average Price (Rs.)", "steelblue");
            renderMetricChart("avg-rating-chart", data, "avg_rating", "Average Rating", "green");
            renderMetricChart("avg-discount-chart", data, "avg_discount", "Average Discount (%)", "orange");
        })
        .catch(error => {
            console.error("Error updating metric charts:", error);
            displayNoDataMessage("avg-price-chart");
            displayNoDataMessage("avg-rating-chart");
            displayNoDataMessage("avg-discount-chart");
        });
    
    // Fetch and render spec frequency data
    if (category !== 'all') {
        const specName = categoryToSpecMap[category] || "ram"; // Default to RAM if not found
        getSpecFrequency(category, specName)
            .then(data => {
                console.log("Spec frequency data received:", data);
                renderSpecFrequencyChart("spec-frequency-chart", data, specName);
            })
            .catch(error => {
                console.error("Error updating spec frequency chart:", error);
                displayNoDataMessage("spec-frequency-chart");
            });
    } else {
        // If "All Categories" is selected, show a message for spec frequency
        displayNoDataMessage("spec-frequency-chart", "Please select a specific category to view spec frequency");
    }
}

// Fetch data for Category Comparison (Avg Price, Rating, Discount by Category)
function getCategoryComparison() {
    const url = 'http://localhost:8000/api/analytics/category-comparison';
    
    console.log("Fetching category comparison data from:", url);
    
    return fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .catch(error => {
            console.error("Error fetching category comparison data:", error);
            throw error; // Re-throw to handle in the calling function
        });
}

// Fetch data for Spec Frequency in a Category
function getSpecFrequency(category, specName) {
    const url = `http://localhost:8000/api/compare/spec-frequency?category=${category}&spec_name=${encodeURIComponent(specName)}`;
    
    console.log("Fetching spec frequency data from:", url);
    
    return fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .catch(error => {
            console.error("Error fetching spec frequency data:", error);
            throw error; // Re-throw to handle in the calling function
        });
}

// Render chart for a specific metric (price, rating, discount)
function renderMetricChart(svgId, data, metricKey, yAxisLabel, barColor) {
    console.log(`Rendering ${metricKey} chart with data:`, data);
    
    if (!data || data.length === 0) {
        console.warn(`No data available for ${metricKey} chart`);
        displayNoDataMessage(svgId);
        return;
    }
    
    const svg = d3.select(`#${svgId}`);
    const width = +svg.attr("width");
    const height = +svg.attr("height");
    const margin = { top: 30, right: 30, bottom: 70, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Clear previous chart
    svg.selectAll("*").remove();

    // Create scales
    const x = d3.scaleBand()
        .domain(data.map(d => d.category))
        .range([0, innerWidth])
        .padding(0.2);

    const y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d[metricKey]) * 1.1]) // Add 10% padding at top
        .range([innerHeight, 0]);

    // Create chart group
    const g = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Add bars
    g.selectAll(".bar")
        .data(data)
        .enter()
        .append("rect")
        .attr("class", "bar")
        .attr("x", d => x(d.category))
        .attr("y", d => y(d[metricKey] || 0)) // Handle null/undefined values
        .attr("width", x.bandwidth())
        .attr("height", d => innerHeight - y(d[metricKey] || 0)) // Handle null/undefined values
        .attr("fill", barColor)
        .on("mouseover", function(event, d) {
            // Show tooltip on hover
            d3.select(this).attr("fill", d3.color(barColor).darker(0.5));
            
            g.append("text")
                .attr("class", "value-label")
                .attr("x", x(d.category) + x.bandwidth() / 2)
                .attr("y", y(d[metricKey] || 0) - 5)
                .attr("text-anchor", "middle")
                .text((d[metricKey] || 0).toFixed(2));
        })
        .on("mouseout", function() {
            d3.select(this).attr("fill", barColor);
            g.selectAll(".value-label").remove();
        });

    // Add x-axis
    g.append("g")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x))
        .selectAll("text")
        .attr("transform", "rotate(-45)")
        .style("text-anchor", "end")
        .attr("dx", "-.8em")
        .attr("dy", ".15em");

    // Add y-axis
    g.append("g")
        .call(d3.axisLeft(y));

    // Add y-axis label
    g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", -margin.left + 20)
        .attr("x", -innerHeight / 2)
        .attr("text-anchor", "middle")
        .text(yAxisLabel);
}

// Render Specs Frequency Chart
function renderSpecFrequencyChart(svgId, data, specName) {
    console.log("Rendering spec frequency chart with data:", data);
    
    if (!data || data.length === 0) {
        console.warn("No data available for spec frequency chart");
        displayNoDataMessage(svgId);
        return;
    }
    
    const svg = d3.select(`#${svgId}`);
    const width = +svg.attr("width");
    const height = +svg.attr("height");
    const margin = { top: 30, right: 30, bottom: 120, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Clear previous chart
    svg.selectAll("*").remove();
    
    // Sort data by frequency (count) in descending order and take top 15
    const sortedData = [...data]
        .sort((a, b) => b.count - a.count)
        .slice(0, 15);

    // Create scales
    const x = d3.scaleBand()
        .domain(sortedData.map(d => d.spec_value))
        .range([0, innerWidth])
        .padding(0.2);

    const y = d3.scaleLinear()
        .domain([0, d3.max(sortedData, d => d.count) * 1.1])
        .range([innerHeight, 0]);

    // Create color scale
    const colorScale = d3.scaleOrdinal()
        .domain(sortedData.map(d => d.spec_value))
        .range(d3.schemeCategory10);

    // Create chart group
    const g = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Add title
    g.append("text")
        .attr("x", innerWidth / 2)
        .attr("y", -10)
        .attr("text-anchor", "middle")
        .style("font-size", "14px")
        .style("font-weight", "bold")
        .text(`Frequency of ${specName}`);

    // Add bars
    g.selectAll(".bar")
        .data(sortedData)
        .enter()
        .append("rect")
        .attr("class", "bar")
        .attr("x", d => x(d.spec_value))
        .attr("y", d => y(d.count))
        .attr("width", x.bandwidth())
        .attr("height", d => innerHeight - y(d.count))
        .attr("fill", d => colorScale(d.spec_value))
        .on("mouseover", function(event, d) {
            d3.select(this).attr("fill", d3.color(colorScale(d.spec_value)).darker(0.5));
            
            g.append("text")
                .attr("class", "value-label")
                .attr("x", x(d.spec_value) + x.bandwidth() / 2)
                .attr("y", y(d.count) - 5)
                .attr("text-anchor", "middle")
                .text(d.count);
        })
        .on("mouseout", function(event, d) {
            d3.select(this).attr("fill", colorScale(d.spec_value));
            g.selectAll(".value-label").remove();
        });

    // Add x-axis
    g.append("g")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x))
        .selectAll("text")
        .attr("transform", "rotate(-45)")
        .style("text-anchor", "end")
        .attr("dx", "-.8em")
        .attr("dy", ".15em");

    // Add y-axis
    g.append("g")
        .call(d3.axisLeft(y));

    // Add y-axis label
    g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", -margin.left + 20)
        .attr("x", -innerHeight / 2)
        .attr("text-anchor", "middle")
        .text("Frequency");
}

// Function to display a "No data available" message
function displayNoDataMessage(svgId, message = "No data available") {
    const svg = d3.select(`#${svgId}`);
    const width = +svg.attr("width");
    const height = +svg.attr("height");
    
    // Clear previous content
    svg.selectAll("*").remove();
    
    // Add message
    svg.append("text")
        .attr("x", width / 2)
        .attr("y", height / 2)
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .style("fill", "#666")
        .text(message);
}

// Initialize charts with default category
document.addEventListener('DOMContentLoaded', function() {
    // Set default category to laptops
    categoryDropdown.value = "laptops";
    selectedCategory = "laptops";
    console.log("DOM loaded, initializing charts with category:", selectedCategory);
    updateAllCharts(selectedCategory);
});
