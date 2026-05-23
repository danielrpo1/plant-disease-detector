"""
Recomendaciones agrícolas de bajo costo por clase detectada.

Orientación general para agricultores pequeños — no sustituye asesoría de un agrónomo.
"""

from __future__ import annotations

TREATMENT_TIPS_ES: dict[str, str] = {
  # Manzana
  "Apple___Apple_scab": (
    "Retira y quema hojas con manchas aceitosas. Mejora ventilación (poda ligera). "
    "Aplica caldo bordelés o bicarbonato de sodio (1 cucharada por litro de agua) cada 7–10 días en época húmeda. "
    "Evita regar las hojas por la noche."
  ),
  "Apple___Black_rot": (
    "Elimina frutos y ramas con manchas negras; desinfecta tijeras con alcohol. "
    "No dejes residuos en el suelo. Refuerza con infusión de ajo o cola de caballo diluida como preventivo."
  ),
  "Apple___Cedar_apple_rust": (
    "Quita hojas amarillas con manchas naranjas. Si hay cedros cerca, poda ramas muy infectadas en ellos. "
    "Aplicaciones de azufre mojable o bicarbonato en primavera reducen propagación."
  ),
  "Apple___healthy": (
    "Planta sana: mantén riego constante sin encharcar, abono orgánico 2–3 veces al año y revisión mensual de hojas."
  ),
  # Arándano
  "Blueberry___healthy": (
    "Sano: suelo ácido (puedes usar agujas de pino compostadas), riego frecuente y sombra parcial en horas muy fuertes."
  ),
  # Cereza
  "Cherry_(including_sour)___Powdery_mildew": (
    "Quita brotes con polvo blanco. No regar por aspersión en hoja. "
    "Leche diluida (1 parte leche + 9 partes agua) o bicarbonato semanal en clima seco ayuda a frenar el hongo."
  ),
  "Cherry_(including_sour)___healthy": (
    "Sano: poda para dejar copa aireada y evita humedad excesiva en el follaje."
  ),
  # Maíz
  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (
    "Rota cultivo (no maíz en el mismo sitio 1–2 temporadas). Elimina restos de la cosecha. "
    "Semillas tratadas o de buena procedencia; evita siembra muy densa."
  ),
  "Corn_(maize)___Common_rust_": (
    "Usa variedades resistentes si puedes. Retira hojas muy infectadas. "
    "Azufre en polvo o caldo bordelés temprano en la temporada; mejora drenaje del suelo."
  ),
  "Corn_(maize)___Northern_Leaf_Blight": (
    "Destino de restos de planta, espacia surcos para ventilación. "
    "Evita exceso de nitrógeno; compost maduro en lugar de químicos costosos."
  ),
  "Corn_(maize)___healthy": (
    "Sano: control de malezas, riego en floración y monitoreo de plagas (trips, gusano cogollero)."
  ),
  # Uva
  "Grape___Black_rot": (
    "Poda sanitaria; retira bayas y hojas mummificadas. Mejora aireación del canopy. "
    "Caldo bordelés tras poda y antes de lluvias prolongadas."
  ),
  "Grape___Esca_(Black_Measles)": (
    "Enfermedad difícil: poda en secano, desinfecta cortes, elimina troncos muy afectados. "
    "Evita heridas grandes; sella cortes con pasta mastic o cera."
  ),
  "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (
    "Retira hojas manchadas del suelo. Riego por goteo en raíz, no mojar follaje. "
    "Infusión de equinácea o cola de caballo como refuerzo preventivo orgánico."
  ),
  "Grape___healthy": (
    "Sano: poda anual, riego moderado y control de densidad de racimos."
  ),
  # Naranja
  "Orange___Haunglongbing_(Citrus_greening)": (
    "No tiene cura definitiva: elimina árboles muy sintomáticos para proteger el huerto. "
    "Controla mosca blanca (neem diluido, trampas amarillas). Solo planta material certificado."
  ),
  # Durazno
  "Peach___Bacterial_spot": (
    "Evita podar con lluvia; desinfecta herramientas. Aplica cobre (caldo cúprico) en dormancia. "
    "Riego por goteo; no trabajes plantas mojadas."
  ),
  "Peach___healthy": (
    "Sano: poda abierta en invierno y eliminación de frutos dañados en verano."
  ),
  # Pimiento
  "Pepper,_bell___Bacterial_spot": (
    "Rota cultivos; no pimiento/tomate en la misma cama seguido. Semilla sana. "
    "Cobre en hojas jóvenes si aparecen manchas; riego en la base."
  ),
  "Pepper,_bell___healthy": (
    "Sano: mulching (paja o plástico) conserva humedad y reduce salpicadura de suelo en hojas."
  ),
  # Papa
  "Potato___Early_blight": (
    "Rota cultivo 3–4 años. Quita hojas basales infectadas. "
    "Bicarbonato o caldo bordelés cada 10 días si hay humedad. Usa papa semilla sana."
  ),
  "Potato___Late_blight": (
    "Urgente en clima húmedo: destruye plantas muy infectadas lejos del cultivo. "
    "No regar hojas; aplica caldo bordelés preventivo antes de lluvias. Evita papa de vecinos infectada."
  ),
  "Potato___healthy": (
    "Sano: aporque (tierra al tallo) y riego regular sin encharcar."
  ),
  # Frambuesa / soja
  "Raspberry___healthy": (
    "Sano: poda de cañas viejas después de cosechar; riego en raíz."
  ),
  "Soybean___healthy": (
    "Sano: inoculación con rizobio (mejora fijación de nitrógeno) y control de malezas temprano."
  ),
  # Calabaza
  "Squash___Powdery_mildew": (
    "Retira hojas blancas; no regar por aspersión. Leche diluida o bicarbonato cada semana. "
    "Deja espacio entre plantas para que circule aire."
  ),
  # Fresa
  "Strawberry___Leaf_scorch": (
    "Riego por goteo; evita sol extremo sin mulching. Retira hojas secas. "
    "No trasplantes de plantas dudosas; solariza suelo antes de nueva siembra."
  ),
  "Strawberry___healthy": (
    "Sano: mulching con paja, riego en raíz y renovar matas cada 2–3 años."
  ),
  # Tomate
  "Tomato___Bacterial_spot": (
    "No tocar plantas mojadas; rota con cultivos no solanáceos. Semilla o plántula certificada. "
    "Cobre preventivo; elimina hojas con manchas negras pequeñas."
  ),
  "Tomato___Early_blight": (
    "Retira hojas inferiores manchadas (anillos concéntricos). Mulching en base. "
    "Caldo bordelés o bicarbonato; riego en la mañana para que sequen las hojas."
  ),
  "Tomato___Late_blight": (
    "Muy contagioso: aísla plantas afectadas, destruye restos. No regar follaje. "
    "Caldo bordelés cada 5–7 días en época lluviosa; mejora ventilación entre plantas."
  ),
  "Tomato___Leaf_Mold": (
    "Ventilación en invernadero o malla; reduce humedad. Riego en suelo. "
    "Retira hojas amarillas del envés con moho verde; bicarbonato ayuda."
  ),
  "Tomato___Septoria_leaf_spot": (
    "Elimina hojas con puntos marrones y halo gris. Mulching; no regar hojas. "
    "Rotación de cultivo; evita trabajar con plantas húmedas."
  ),
  "Tomato___Spider_mites Two-spotted_spider_mite": (
    "Ácaros: enjuaga hojas con agua a presión suave. Jabón potásico o aceite de neem (5 ml/L) "
    "cada 3 días bajo las hojas. Aumenta humedad ambiental sin encharcar raíz."
  ),
  "Tomato___Target_Spot": (
    "Retira hojas infectadas; mulching. Evita estrés por sequía. "
    "Bicarbonato o extracto de ajo; espacia plantas."
  ),
  "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
    "Virus transmitido por mosca blanca: trampas amarillas, neem en hojas. "
    "Arranca plantas muy enfermas. No reutilices tutores sin lavar; controla malezas hospederas."
  ),
  "Tomato___Tomato_mosaic_virus": (
    "No hay cura: elimina plantas con mosaico verde-amarillo. Lava manos antes de podar. "
    "No fumes cerca del cultivo (se transmite); usa variedades tolerantes si existen."
  ),
  "Tomato___healthy": (
    "Sano: tutorado, riego regular en la base y revisión semanal de plagas."
  ),
}

DISCLAIMER_ES = (
  "Estas recomendaciones son orientación general de manejo con insumos de bajo costo. "
  "Para cultivos comerciales o brotes graves, consulte un agrónomo o servicio de extensión local."
)


def treatment_tip(class_id: str) -> str:
    if "healthy" in class_id.lower():
        return TREATMENT_TIPS_ES.get(class_id, "Planta sin signos evidentes: siga con buenas prácticas de riego, poda y limpieza.")
    return TREATMENT_TIPS_ES.get(
        class_id,
        "Retire tejido muy dañado, mejore ventilación, evite regar el follaje y consulte extensión agrícola local.",
    )
