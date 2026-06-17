function change(fieldId,delta){
  const input = document.getElementById(fieldId);
  const min = parseInt(input.min)||0;
  const max= parseInt(input.max)|| 99;
  const val = parseInt(input.value) ||0;
  input.value = Math.min(max, Math.max(min, val + delta));
}
//  rupess functionality format
function formatINR(amount) {
  return "₹" + amount.toLocaleString("en-IN");
}
function showError(msg) {
  const el = document.getElementById("error-msg");
  el.textContent = " " + msg;
  el.style.display = "block";
}
function clearError() {
  const el = document.getElementById("error-msg");
  el.style.display = "none";
  el.textContent   = "";
}
async function predict() {
  clearError(); 
  const sqft = document.getElementById("sqft").value.trim();
  const bedrooms = document.getElementById("bedrooms").value;
  const full_bath =document.getElementById("full_bath").value;
  const half_bath= document.getElementById("half_bath").value;

  if (!sqft || sqft <= 0) {
    showError("Please enter a valid total area.");
    return;
  } 
  const btn = document.getElementById("predict-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Calculating…';
  try {
    const formData = new FormData();
        formData.append("sqft", sqft);
        formData.append("bedrooms",bedrooms);
        formData.append("full_bath",full_bath);
        formData.append("half_bath",half_bath);
    const response = await fetch("/predict",{method:"POST",body:formData});
    const data = await response.json();
 
    if (data.error) {
      showError(data.error);
      return;
    }
    document.getElementById("result-price").textContent =formatINR(data.price);
    document.getElementById("result-range").textContent =
      `Range: ${formatINR(data.range_low)} – ${formatINR(data.range_high)}`;
    const totalBath = parseFloat(full_bath) + parseFloat(half_bath) * 0.5;
    document.getElementById("s-sqft").textContent = parseInt(sqft).toLocaleString("en-IN");
    document.getElementById("s-beds").textContent = bedrooms;
    document.getElementById("s-baths").textContent = totalBath;

    document.getElementById("result-placeholder").style.display = "none";
    const content = document.getElementById("result-content");
    content.style.display = "block";
    content.classList.add("fade-in");
 
  } catch (err) {
    showError("Cannot reach server. Make sure Flask is running.");
  } finally{
    btn.disabled  = false;
    btn.innerHTML = "Get Price Estimate";
  }
}

function resetForm(){
  document.getElementById("sqft").value= "";
  document.getElementById("bedrooms").value = "2";
  document.getElementById("full_bath").value = "1";
  document.getElementById("half_bath").value = "0";
  clearError();
  document.getElementById("result-placeholder").style.display = "block";
  const content = document.getElementById("result-content");
  content.style.display = "block";
  content.classList.remove("fade-in");
}
//  enter key entry 
document.addEventListener("keydown", function(e) {
  if (e.key === "Enter") predict();
});
