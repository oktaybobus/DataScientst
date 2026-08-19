import streamlit as st

from train_codes import TRAIN_CODE
from components import show_dataset_info, show_model_metrics


def render():
    project = st.sidebar.selectbox("Select an Agent Scenario", [
        "Smart Farming & Irrigation Agent",
        "FAQ Support & Intent Analysis Agent",
        "Autonomous Vehicle Simulation Agent"
    ])

    # ----------------- 1. SMART FARMING & IRRIGATION AGENT -----------------
    if project == "Smart Farming & Irrigation Agent":
        st.header("🌾 Smart Farming & Irrigation Decision Agent")
        st.write("The agent monitors soil moisture, temperature, and sunlight data in real-time to make autonomous decisions that prevent water waste.")

        show_dataset_info("farm_agent")

        with st.expander("⚙️ View Agent Decision Logic"):
            st.code(TRAIN_CODE["farm_agent"], language="python")

        show_model_metrics("farm_agent")

        col1, col2 = st.columns(2)
        with col1:
            soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 25)
            temperature = st.slider("Air Temperature (°C)", -10, 50, 32)
            sunlight = st.select_slider("Sunlight Intensity", options=["Low", "Medium", "High"])

        with col2:
            st.subheader("🤖 Agent's Autonomous Decision Mechanism")
            if st.button("Analyze Sensor Data"):
                st.info("🔄 The farming agent is reading sensor logs and calculating evaporation risk...")

                # Otonom Karar Ağacı Mantığı
                if soil_moisture < 30 and temperature > 30 and sunlight == "High":
                    st.error("🚨 DECISION: Urgent Heavy Irrigation Required! (High evaporation risk, soil is critically dry).")
                    st.metric("Recommended Water Amount", "45 Liters / m²")
                elif soil_moisture < 40 and temperature > 15:
                    st.warning("⚠️ DECISION: Standard Irrigation Started. (Soil moisture is below ideal threshold).")
                    st.metric("Recommended Water Amount", "20 Liters / m²")
                else:
                    st.success("✅ DECISION: No Irrigation Needed. (Soil moisture is sufficient, water-saving mode active).")
                    st.metric("Recommended Water Amount", "0 Liters / m²")

    # ----------------- 2. FAQ SUPPORT & INTENT ANALYSIS AGENT -----------------
    elif project == "FAQ Support & Intent Analysis Agent":
        st.header("💬 FAQ Support & Intent Analysis Agent")
        st.write("The corporate customer agent autonomously analyzes the intent behind incoming messages and triggers the appropriate action.")

        show_dataset_info("faq_agent")

        with st.expander("⚙️ View Agent Decision Logic"):
            st.code(TRAIN_CODE["faq_agent"], language="python")

        show_model_metrics("faq_agent")

        user_message = st.text_area("Customer Message (you can modify the sample text):",
                                    "Hello, I placed an order 3 days ago and it still hasn't been shipped. I want to cancel and get a refund.")

        if st.button("Process Message with Agent"):
            st.info("🔄 The agent is performing intent classification and keyword scanning...")

            # Basit kural tabanlı niyet yakalama simülasyonu
            msg_lower = user_message.lower()

            if "refund" in msg_lower or "cancel" in msg_lower or "money" in msg_lower:
                intent = "🚨 Finance / Refund and Cancellation Request"
                action = "Customer's invoice history verified. Cancellation request autonomously forwarded to the Accounting department and refund process initiated."
            elif "shipping" in msg_lower or "order" in msg_lower or "where" in msg_lower or "shipped" in msg_lower:
                intent = "📦 Logistics / Shipping and Delivery Tracking"
                action = "Attempting to locate order number. The system autonomously sent a tracking query to the shipping carrier API."
            else:
                intent = "💬 General / Thank You - Information Request"
                action = "Message routed to the standard FAQ pool. The agent is preparing an automated AI response."

            st.subheader("🤖 Agent Analysis Report:")
            st.write(f"**Detected Intent:** {intent}")
            st.success(f"**Autonomous Action Taken:** {action}")

    # ----------------- 3. AUTONOMOUS VEHICLE SIMULATION AGENT -----------------
    elif project == "Autonomous Vehicle Simulation Agent":
        st.header("🚗 Autonomous Vehicle Lane Tracking & Parking Agent")
        st.write("The virtual vehicle agent evaluates sensor inputs to simulate lane-keeping and autonomous parking decisions.")

        show_dataset_info("autonomous_car")

        with st.expander("⚙️ View Agent Decision Logic"):
            st.code(TRAIN_CODE["autonomous_car"], language="python")

        show_model_metrics("autonomous_car")

        sensor_distance = st.slider("Distance to Front Vehicle (Meters)", 1, 100, 15)
        lane_status = st.selectbox("Lane Marking Status", ["Clearly Visible", "Dashed / Faded", "No Lane Markings"])
        parking_slot = st.checkbox("Empty Parking Spot Detected?")

        if st.button("Run Vehicle Agent"):
            st.subheader("🎬 Agent Decisions During Driving:")

            # Şerit takip kararı
            if lane_status == "Clearly Visible":
                st.success("🟢 LANE TRACKING: Cameras active. Autonomous driving safely continues with centered lane positioning.")
            else:
                st.warning("🟡 LANE WARNING: Lane markings insufficient! Agent is preparing to hand over steering control to the driver.")

            # Mesafe ve Fren kararı
            if sensor_distance < 20:
                st.error(f"🔴 EMERGENCY BRAKE: Distance to front vehicle dropped to {sensor_distance} meters! Autonomous braking applied due to safe following distance violation.")
            else:
                st.success("🟢 SPEED CONTROL: Distance is safe. Autonomous driving continues at the set speed limit.")

            # Park Kararı
            if parking_slot and sensor_distance > 30:
                st.info("🔵 AUTONOMOUS PARKING: Empty parking spot detected. Vehicle speed is being reduced and the autonomous perpendicular parking algorithm is starting.")
