const { rules, triggers, items, actions } = require('openhab');

let lastAlertTimestamp = 0;
let lastAlertSeverity = "NORMAL";
const COOLDOWN_MS = 10 * 60 * 1000;

rules.JSRule({
  name: "Edge AI Multi-Variate Anomaly Detection",
  description: "Запускає Python ML інференс при оновленні телеметрії та керує тривогами",
  triggers: [
    triggers.GroupStateChangeTrigger("gTelemetry")
  ],
  execute: (event) => {
    const powerItem = items.getItem("Device_Power_Watts");
    const tempItem = items.getItem("Room_Temperature");
    const humItem = items.getItem("Room_Humidity");

    if (powerItem.isUninitialized || tempItem.isUninitialized || humItem.isUninitialized) {
      return;
    }

    const power = parseFloat(powerItem.state);
    const temp = parseFloat(tempItem.state);
    const humidity = parseFloat(humItem.state);

    const scriptPath = "/openhab/conf/scripts/infer_anomaly.py";
    const rawOutput = actions.Exec.executeCommandLine(
      time.Duration.ofSeconds(5),
      "python3",
      scriptPath,
      "--power", power.toString(),
      "--temp", temp.toString(),
      "--humidity", humidity.toString()
    );

    if (!rawOutput) {
      console.error("[EdgeAI] Порожня відповідь від інференс-скрипту.");
      return;
    }

    try {
      const result = JSON.parse(rawOutput.trim());

      items.getItem("AI_Anomaly_Score").postUpdate(result.anomaly_score);
      items.getItem("AI_Adaptive_Threshold").postUpdate(result.threshold);
      items.getItem("AI_Anomaly_Severity").postUpdate(result.severity);
      items.getItem("AI_Anomaly_Explanation").postUpdate(result.explanation);

      if (result.is_anomaly) {
        items.getItem("AI_Anomaly_Alert").sendCommand("ON");

        const now = Date.now();
        const isEscalated = (result.severity === "CRITICAL" && lastAlertSeverity !== "CRITICAL");

        if ((now - lastAlertTimestamp > COOLDOWN_MS) || isEscalated) {
          console.warn(`[EdgeAI ALERT] ${result.explanation}`);
          lastAlertTimestamp = now;
          lastAlertSeverity = result.severity;
        }
      } else {
        items.getItem("AI_Anomaly_Alert").sendCommand("OFF");
        lastAlertSeverity = "NORMAL";
      }

      if (result.drift_detected) {
        console.warn("[EdgeAI DRIFT] Виявлено зміщення статистичного розподілу! Запустіть train_model.py.");
      }

    } catch (e) {
      console.error(`[EdgeAI] Помилка парсингу JSON: ${e.message} | Вивід: ${rawOutput}`);
    }
  }
});
