import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import time
import os
import hashlib
import re
from fpdf import FPDF
import io
import smtplib
from email.message import EmailMessage
from datetime import datetime
import streamlit.components.v1 as components
import stripe

# Ocultar el menú de Streamlit y el pie de página
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.set_page_config(page_title="Spacenet AI | Control de Misiones", layout="wide")

# ==========================================
# PLANES Y PRECIOS (EUR / céntimos Stripe)
# ==========================================
PLANES = {
    "Starter": {
        "precio_eur": 79,
        "precio_centimos": 7900,
        "descripcion": "Purificación básica y exportación CSV",
    },
    "Pro": {
        "precio_eur": 199,
        "precio_centimos": 19900,
        "descripcion": "Informe PDF premium, datasets IA y soporte prioritario",
    },
    "Business": {
        "precio_eur": 499,
        "precio_centimos": 49900,
        "descripcion": "Multi-equipo, volumen alto y acompañamiento dedicado",
    },
}

# Precios one-shot del marketplace de datasets limpios (data flywheel)
PRECIO_DATASET_PRO_EUR = 49
PRECIO_DATASET_BUSINESS_EUR = 99
PLANES_PRO_O_SUPERIOR = {"Pro", "Business"}
MSG_PDF_REQUIERE_PRO = (
    "Este módulo requiere suscripción Pro o superior activa. "
    "Por favor renueva tu suscripción para continuar."
)
MSG_PDF_PREMIUM = "📄 PDF premium requiere suscripción activa"

# ==========================================
# AUTHENTICATION (MULTI-TENANT)
# ==========================================
def verificar_credenciales(usuario, password):
    try:
        usuarios_db = st.secrets["usuarios"]
        if usuario in usuarios_db:
            if str(usuarios_db[usuario]["password"]) == str(password):
                return True, usuarios_db[usuario]["empresa"]
    except Exception as e:
        st.error(f"Error de configuración en Secrets: {e}")
    return False, None

if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False
    st.session_state["empresa"] = None

if not st.session_state["autenticado"]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("Acceso al Sistema B2B")
        user_input = st.text_input("Usuario Corporativo")
        pass_input = st.text_input("Contraseña", type="password")
        if st.button("Ingresar"):
            valido, empresa_cliente = verificar_credenciales(user_input, pass_input)
            if valido:
                st.session_state["autenticado"] = True
                st.session_state["empresa"] = empresa_cliente
                st.rerun()
            else:
                st.error("Credenciales inválidas. Contacte a soporte de Spacenet.")
    st.stop()

# ==========================================
# CORE DE DATOS Y AI
# ==========================================
DATABASE_URL = st.secrets["DATABASE_URL"]

def obtener_conexion():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

@st.cache_resource
def cargar_cerebro_ia():
    if os.path.exists('autoencoder_pesos.weights.h5') and os.path.exists('escalador_ia.pkl'):
        input_layer = layers.Input(shape=(4,))
        encoded = layers.Dense(8, activation='relu')(input_layer)
        bottleneck = layers.Dense(3, activation='relu')(encoded)
        decoded = layers.Dense(8, activation='relu')(bottleneck)
        output_layer = layers.Dense(4, activation='sigmoid')(decoded)
        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.load_weights('autoencoder_pesos.weights.h5')
        scaler = joblib.load('escalador_ia.pkl')
        return model, scaler
    return None, None

autoencoder, scaler = cargar_cerebro_ia()

def anonimizar_texto(texto):
    return hashlib.sha256(str(texto).encode('utf-8')).hexdigest()[:12]

def blindar_telefono(telefono):
    tel_str = str(telefono).strip()
    if len(tel_str) < 7:
        return "*******"
    return f"{tel_str[:3]} ****** {tel_str[-3:]}"

def adaptar_columnas_tenant(df, empresa_actual):
    """Normaliza columnas específicas de cada tenant al esquema del autoencoder."""
    df = df.copy()
    if empresa_actual == "Agencia_X":
        df = df.rename(columns={
            "Presupuesto_Mensual": "Ingresos_Anuales",
            "Años": "Edad",
            "Correo": "Email",
            "Cliente_Nombre": "Nombre",
        })
    elif empresa_actual == "Ecommerce_Y":
        df = df.rename(columns={
            "Facturacion": "Ingresos_Anuales",
            "Email_Contacto": "Email",
        })
    return df

def purificar_datos_con_ia(df_sucio):
    df_limpio = df_sucio.copy()
    df_limpio['Edad'] = df_limpio['Edad'].fillna(df_limpio['Edad'].median() if not df_limpio['Edad'].dropna().empty else 40)
    df_limpio['Ingresos_Anuales'] = df_limpio['Ingresos_Anuales'].fillna(df_limpio['Ingresos_Anuales'].median())
    df_limpio['Email_Roto'] = df_limpio['Email'].apply(lambda x: 1.0 if pd.isna(x) or '@' not in str(x) else 0.0)
    df_limpio['Nombre_Falso'] = df_limpio['Nombre'].apply(lambda x: 1.0 if bool(re.search(r'\d', str(x))) else 0.0)

    X_nuevos = df_limpio[['Edad', 'Ingresos_Anuales', 'Email_Roto', 'Nombre_Falso']].values
    x_min, x_max = scaler
    X_nuevos_scaled = (X_nuevos - x_min) / (x_max - x_min + 1e-5)

    X_reconstruido = autoencoder.predict(X_nuevos_scaled, verbose=0)
    errores_reconstruccion = np.mean(np.power(X_nuevos_scaled - X_reconstruido, 2), axis=1)

    UMBRAL_IA = 0.05
    df_sucio = df_sucio.copy()
    df_sucio['Error_IA'] = errores_reconstruccion
    df_aprobado = df_limpio[errores_reconstruccion <= UMBRAL_IA].copy()

    df_aprobado['Nombre'] = df_aprobado['Nombre'].apply(anonimizar_texto)
    df_aprobado['Email'] = df_aprobado['Email'].apply(anonimizar_texto)
    df_aprobado['Telefono'] = df_aprobado['Telefono'].apply(blindar_telefono)

    return df_aprobado, len(df_sucio), df_sucio['Edad'].isnull().sum(), len(df_sucio[errores_reconstruccion > UMBRAL_IA]), df_sucio

# ==========================================
# GENERADOR DE REPORTES (PDF PREMIUM)
# ==========================================
def generar_reporte_pdf(total, nulos, alertas, empresa):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 38, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 22)
    pdf.cell(0, 10, "SPACENET DATA INTELLIGENCE", ln=True, align='L')
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5, "AI-Powered Data Purification & Security Audit", ln=True, align='L')
    pdf.ln(20)

    pdf.set_text_color(30, 41, 59)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "IMPACTO DE NEGOCIO ESTIMADO", ln=True)
    pdf.set_fill_color(240, 253, 244)
    pdf.rect(10, pdf.get_y(), 190, 20, 'F')
    pdf.set_text_color(22, 101, 52)
    pdf.set_font("Arial", size=11)
    ahorro_estimado = alertas * 15
    pdf.multi_cell(0, 8, f"  Al bloquear {alertas} anomalias, hemos evitado un posible coste operativo o de fraude\n  estimado en {ahorro_estimado} EUR para su operacion actual.")
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "METRICAS CLAVE DE PURIFICACION", ln=True)
    pdf.ln(2)

    pdf.set_fill_color(241, 245, 249)
    pdf.set_text_color(71, 85, 105)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(110, 10, " Indicador Analizado", border=1, fill=True)
    pdf.cell(80, 10, " Valor / Estado", border=1, fill=True, align='C')
    pdf.ln()

    pdf.cell(0, 10, f"Coste estimado de ineficiencia: {alertas * 0.50} EUR", ln=True)

    data = [
        ("Empresa / Tenant", str(empresa)),
        ("Registros Totales Auditados", f"{total:,} filas"),
        ("Registros Nulos Corregidos por IA", f"{nulos:,} correcciones"),
        ("Anomalias / Fraudes Detectados (Bloqueados)", f"{alertas:,} alertas"),
        ("Estado Final del Dataset", "PURIFICADO & SEGURO"),
        ("Protocolo de Criptografia Aplicado", "SHA-256 + Phone Masking"),
    ]

    pdf.set_font("Arial", size=10)
    for label, value in data:
        if "Anomalias" in label and alertas > 0:
            pdf.set_text_color(220, 38, 38)
            pdf.set_font("Arial", 'B', 10)
        else:
            pdf.set_text_color(51, 65, 85)
            pdf.set_font("Arial", size=10)

        pdf.cell(110, 9, f"  {label}", border=1)
        pdf.cell(80, 9, f" {value}", border=1, align='C')
        pdf.ln()

    pdf.ln(12)
    pdf.set_fill_color(239, 246, 255)
    pdf.set_draw_color(191, 219, 254)
    pdf.rect(10, pdf.get_y(), 190, 25, 'DF')

    pdf.set_text_color(29, 78, 216)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, "  DIAGNOSTICO DEL INGENIERO DE IA:", ln=True)
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(30, 41, 59)

    if alertas > 0:
        msg_diagnostico = (
            f"  ATENCION: Se han detectado {alertas} vectores de riesgo en el dataset. Los datos han sido aislados\n"
            "  en la sala de cuarentena para proteger la integridad de su base de datos corporativa."
        )
    else:
        msg_diagnostico = (
            "  OPTIMO: No se han detectado anomalias criticas. El dataset cumple con los estandares internacionales\n"
            "  de calidad y politicas de privacidad de datos."
        )

    pdf.multi_cell(0, 5, msg_diagnostico)

    pdf.set_y(-25)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 10, "CONFIDENCIAL - Spacenet AI Engine v2.0 - Copia de Seguridad Autorizada", border=0, align='C')

    return bytes(pdf.output())

def enviar_aviso_venta(nombre, email, empresa):
    mi_correo = st.secrets["EMAIL_DESTINO"]
    contrasena = st.secrets["EMAIL_PASSWORD"]

    msg = EmailMessage()
    msg['Subject'] = f"🚀 NUEVA VENTA: {nombre} ({empresa}) ha auditado datos"
    msg['From'] = mi_correo
    msg['To'] = mi_correo
    msg.set_content(f"El cliente {nombre} ({email}) de la empresa {empresa} acaba de descargar un informe.")

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(mi_correo, contrasena)
        smtp.send_message(msg)

def enviar_aviso_pago(nombre, email, empresa, plan, trial=False):
    """Notifica por email un pago o prueba iniciada en Stripe Checkout."""
    mi_correo = st.secrets["EMAIL_DESTINO"]
    contrasena = st.secrets["EMAIL_PASSWORD"]
    tipo = "prueba gratuita" if trial else "pago"
    precio = PLANES.get(plan, {}).get("precio_eur", "?")

    msg = EmailMessage()
    msg['Subject'] = f"💳 NUEVA SUSCRIPCIÓN ({tipo.upper()}): {empresa} → {plan}"
    msg['From'] = mi_correo
    msg['To'] = mi_correo
    msg.set_content(
        f"El cliente {nombre} ({email}) de la empresa {empresa} ha completado Checkout.\n"
        f"Plan: {plan} ({precio}€/mes)\n"
        f"Tipo: {tipo}\n"
        f"Se ha generado el PDF premium y se ha guardado en clientes_purificados."
    )

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(mi_correo, contrasena)
        smtp.send_message(msg)

def enviar_aviso_dataset(asunto, cuerpo):
    """Notifica ventas/listados del marketplace de datasets."""
    mi_correo = st.secrets["EMAIL_DESTINO"]
    contrasena = st.secrets["EMAIL_PASSWORD"]
    msg = EmailMessage()
    msg['Subject'] = asunto
    msg['From'] = mi_correo
    msg['To'] = mi_correo
    msg.set_content(cuerpo)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(mi_correo, contrasena)
        smtp.send_message(msg)

def precio_dataset_segun_plan(plan):
    """49€ (Pro) o 99€ (Business) por dataset limpio."""
    if plan == "Business":
        return PRECIO_DATASET_BUSINESS_EUR
    return PRECIO_DATASET_PRO_EUR

def clave_dataset(empresa, dataset_nombre):
    return f"{empresa}::{dataset_nombre}"

def dataset_ya_comprado(empresa, dataset_nombre):
    comprados = st.session_state.get("datasets_comprados") or set()
    return clave_dataset(empresa, dataset_nombre) in comprados

def marcar_dataset_comprado(empresa, dataset_nombre):
    if "datasets_comprados" not in st.session_state or st.session_state.datasets_comprados is None:
        st.session_state.datasets_comprados = set()
    st.session_state.datasets_comprados.add(clave_dataset(empresa, dataset_nombre))

# ==========================================
# STRIPE CHECKOUT + CUSTOMER PORTAL
# ==========================================
stripe.api_key = st.secrets["STRIPE_SECRET_KEY"]

def obtener_url_base():
    try:
        return str(st.secrets["APP_URL"]).rstrip("/")
    except Exception:
        return "http://localhost:8501"

def obtener_email_facturacion():
    return (st.session_state.get("email_facturacion") or "").strip()

def obtener_o_crear_cliente_stripe(email, empresa):
    email = (email or "").strip()
    if not email or "@" not in email:
        raise ValueError("Email de facturación no válido")

    clientes = stripe.Customer.list(email=email, limit=1)
    if clientes.data:
        cliente = clientes.data[0]
        stripe.Customer.modify(
            cliente.id,
            metadata={"empresa": empresa},
            name=empresa,
        )
        return cliente.id

    cliente = stripe.Customer.create(
        email=email,
        name=empresa,
        metadata={"empresa": empresa},
    )
    return cliente.id

def crear_sesion_checkout(email, empresa, plan="Pro", trial=False):
    if plan not in PLANES:
        raise ValueError(f"Plan desconocido: {plan}")

    info = PLANES[plan]
    cliente_id = obtener_o_crear_cliente_stripe(email, empresa)
    base = obtener_url_base()

    # Permite usar Price IDs de secrets si existen; si no, price_data dinámico
    price_secret_key = f"STRIPE_PRICE_{plan.upper()}"
    try:
        price_id = st.secrets[price_secret_key]
        line_items = [{"price": price_id, "quantity": 1}]
    except Exception:
        line_items = [{
            "price_data": {
                "currency": "eur",
                "unit_amount": info["precio_centimos"],
                "recurring": {"interval": "month"},
                "product_data": {
                    "name": f"Spacenet AI {plan}",
                    "description": info["descripcion"],
                },
            },
            "quantity": 1,
        }]

    params = {
        "mode": "subscription",
        "customer": cliente_id,
        "line_items": line_items,
        "success_url": f"{base}/?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
        "cancel_url": f"{base}/?checkout=cancel",
        "client_reference_id": empresa,
        "metadata": {
            "empresa": empresa,
            "plan": plan,
            "trial": "1" if trial else "0",
        },
        "subscription_data": {
            "metadata": {
                "empresa": empresa,
                "plan": plan,
            },
        },
        "locale": "es",
        "allow_promotion_codes": True,
    }
    if trial:
        params["subscription_data"]["trial_period_days"] = 7

    return stripe.checkout.Session.create(**params)

def abrir_url_stripe(url, etiqueta="Ir a Stripe Checkout →"):
    """Abre Checkout de Stripe (nueva pestaña) con botón de respaldo."""
    st.session_state["stripe_checkout_url"] = url
    try:
        components.html(
            f"""
            <script>
              window.open("{url}", "_blank");
            </script>
            """,
            height=0,
        )
    except Exception:
        pass
    st.link_button(etiqueta, url, type="primary")
    st.caption("Si no se abre sola, pulsa el botón para ir a Stripe Checkout.")

def crear_sesion_checkout_venta_dataset(email, empresa, plan="Pro"):
    """Checkout one-shot al pulsar «Vender Dataset Limpio» (49€ Pro / 99€ Business)."""
    if plan not in PLANES_PRO_O_SUPERIOR:
        plan = "Pro"
    precio_eur = precio_dataset_segun_plan(plan)
    cliente_id = obtener_o_crear_cliente_stripe(email, empresa)
    base = obtener_url_base()

    # Price ID opcional en secrets; si no, price_data dinámico
    try:
        price_id = st.secrets["STRIPE_PRICE_DATASET_BUSINESS" if plan == "Business" else "STRIPE_PRICE_DATASET"]
        line_items = [{"price": price_id, "quantity": 1}]
    except Exception:
        line_items = [{
            "price_data": {
                "currency": "eur",
                "unit_amount": int(precio_eur) * 100,
                "product_data": {
                    "name": f"Venta Dataset Limpio Spacenet AI ({plan})",
                    "description": (
                        f"Purificación + publicación marketplace · "
                        f"{precio_eur}€ · tenant {empresa}"
                    ),
                },
            },
            "quantity": 1,
        }]

    return stripe.checkout.Session.create(
        mode="payment",
        customer=cliente_id,
        line_items=line_items,
        success_url=f"{base}/?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base}/?checkout=cancel",
        client_reference_id=empresa,
        metadata={
            "tipo": "venta_dataset",
            "empresa": empresa,
            "plan": plan,
            "precio_eur": str(precio_eur),
        },
        locale="es",
        allow_promotion_codes=True,
    )

def crear_sesion_checkout_dataset(email, empresa_comprador, empresa_vendedor, dataset_nombre, precio_eur):
    """Checkout one-shot para comprar un dataset limpio del marketplace."""
    cliente_id = obtener_o_crear_cliente_stripe(email, empresa_comprador)
    base = obtener_url_base()
    precio_centimos = int(precio_eur) * 100

    return stripe.checkout.Session.create(
        mode="payment",
        customer=cliente_id,
        line_items=[{
            "price_data": {
                "currency": "eur",
                "unit_amount": precio_centimos,
                "product_data": {
                    "name": f"Dataset limpio: {dataset_nombre}",
                    "description": (
                        f"Dataset purificado por {empresa_vendedor} · "
                        f"listo para entrenar IA · {precio_eur}€"
                    ),
                },
            },
            "quantity": 1,
        }],
        success_url=f"{base}/?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base}/?checkout=cancel",
        client_reference_id=empresa_comprador,
        metadata={
            "tipo": "dataset",
            "empresa": empresa_comprador,
            "empresa_vendedor": empresa_vendedor,
            "dataset_nombre": dataset_nombre,
            "precio_eur": str(precio_eur),
        },
        locale="es",
        allow_promotion_codes=True,
    )

def crear_sesion_portal(email, empresa):
    cliente_id = obtener_o_crear_cliente_stripe(email, empresa)
    base = obtener_url_base()
    return stripe.billing_portal.Session.create(
        customer=cliente_id,
        return_url=f"{base}/",
        locale="es",
    )

def cancelar_suscripcion_stripe(email):
    """Cancela todas las suscripciones activas del cliente (stripe.Subscription.cancel)."""
    email = (email or "").strip()
    if not email:
        return False, "Introduce un email de facturación."

    clientes = stripe.Customer.list(email=email, limit=1)
    if not clientes.data:
        return False, "No se encontró ningún cliente de Stripe con ese email."

    cliente_id = clientes.data[0].id
    suscripciones = stripe.Subscription.list(customer=cliente_id, status="active", limit=20)
    if not suscripciones.data:
        # También cancelar trials
        trials = stripe.Subscription.list(customer=cliente_id, status="trialing", limit=20)
        suscripciones = trials
    if not suscripciones.data:
        return False, "No hay suscripciones activas ni en periodo de prueba."

    canceladas = []
    for sub in suscripciones.data:
        stripe.Subscription.cancel(sub.id)
        canceladas.append(sub.id)

    return True, f"Suscripción cancelada correctamente ({len(canceladas)})."

def _inferir_plan_desde_suscripcion(sub):
    """Obtiene el plan desde metadata Stripe o importe (céntimos)."""
    meta = sub.get("metadata") or {}
    plan = meta.get("plan")
    if plan in PLANES:
        return plan
    try:
        items = sub.get("items", {}).get("data") or []
        if items:
            amount = items[0].get("price", {}).get("unit_amount")
            for nombre, info in PLANES.items():
                if info["precio_centimos"] == amount:
                    return nombre
            product_name = (items[0].get("price", {}).get("product") or "")
            # Si product es id, no ayuda; a veces viene expandido con name vía nickname
            nickname = items[0].get("price", {}).get("nickname") or ""
            for nombre in PLANES:
                if nombre.lower() in str(nickname).lower():
                    return nombre
    except Exception:
        pass
    return None

def obtener_suscripciones_vivas(email):
    """Devuelve suscripciones active/trialing del email (multi-tenant vía metadata empresa)."""
    if not email:
        return []
    clientes = stripe.Customer.list(email=email.strip(), limit=1)
    if not clientes.data:
        return []
    cliente_id = clientes.data[0].id
    activas = stripe.Subscription.list(customer=cliente_id, status="active", limit=10)
    trials = stripe.Subscription.list(customer=cliente_id, status="trialing", limit=10)
    return list(activas.data) + list(trials.data)

def verificar_suscripcion_activa(email):
    """Busca en Stripe si el email tiene una suscripción mensual activa o en trial."""
    if not email:
        return False
    try:
        return len(obtener_suscripciones_vivas(email)) > 0
    except Exception as e:
        st.error(f"Error de conexión con la pasarela de pagos: {e}")
        return False

def sincronizar_plan_desde_stripe(email):
    """Actualiza plan_activo en sesión a partir de Stripe. Devuelve el plan o None."""
    try:
        subs = obtener_suscripciones_vivas(email)
    except Exception:
        return None
    for sub in subs:
        plan = _inferir_plan_desde_suscripcion(sub)
        if plan:
            st.session_state.plan_activo = plan
            st.session_state.pro_unlocked = True
            return plan
    if subs:
        st.session_state.pro_unlocked = True
        return st.session_state.get("plan_activo")
    return None

def es_admin_tenant():
    return st.session_state.get("empresa") == "Spacenet_Admin"

def tiene_suscripcion_pro_o_superior():
    """True si es admin o tiene plan Pro/Business activo (PDF premium)."""
    if es_admin_tenant():
        return True
    if not st.session_state.get("pro_unlocked"):
        return False
    plan = st.session_state.get("plan_activo")
    return plan in PLANES_PRO_O_SUPERIOR

def obtener_metricas_para_pdf(empresa):
    """Usa métricas de sesión si existen; si no, estima desde la DB del tenant."""
    if st.session_state.get("metricas"):
        return st.session_state.metricas

    try:
        conn = obtener_conexion()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM clientes_purificados WHERE empresa = %s",
            (empresa,),
        )
        total = cursor.fetchone()[0] or 0
        cursor.close()
        conn.close()
        return total, 0, 0
    except Exception:
        return 0, 0, 0

def asegurar_columnas_premium():
    """Añade columnas necesarias para PDF premium y datasets vendidos (multi-tenant)."""
    conn = obtener_conexion()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "ALTER TABLE clientes_purificados ADD COLUMN IF NOT EXISTS empresa VARCHAR(100) DEFAULT 'Desconocida'"
        )
        cursor.execute(
            "ALTER TABLE clientes_purificados ADD COLUMN IF NOT EXISTS pdf_informe BYTEA"
        )
        cursor.execute(
            "ALTER TABLE clientes_purificados ADD COLUMN IF NOT EXISTS dataset_vendido BOOLEAN DEFAULT FALSE"
        )
        cursor.execute(
            "ALTER TABLE clientes_purificados ADD COLUMN IF NOT EXISTS dataset_nombre VARCHAR(200)"
        )
        cursor.execute(
            "ALTER TABLE clientes_purificados ADD COLUMN IF NOT EXISTS plan_stripe VARCHAR(50)"
        )
        cursor.execute(
            "ALTER TABLE clientes_purificados ADD COLUMN IF NOT EXISTS fecha_dataset TIMESTAMP"
        )
        cursor.execute(
            "ALTER TABLE clientes_purificados ADD COLUMN IF NOT EXISTS precio_dataset INTEGER"
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def guardar_pdf_en_db(empresa, pdf_bytes, plan):
    """Guarda el PDF premium en clientes_purificados para el tenant actual."""
    asegurar_columnas_premium()
    conn = obtener_conexion()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            UPDATE clientes_purificados
            SET pdf_informe = %s, plan_stripe = %s
            WHERE empresa = %s
            """,
            (psycopg2.Binary(pdf_bytes), plan, empresa),
        )
        if cursor.rowcount == 0:
            # Si aún no hay filas del tenant, insertamos un registro contenedor del PDF
            cursor.execute(
                """
                INSERT INTO clientes_purificados
                    (empresa, ID_Cliente, Nombre, Email, Edad, Ingresos_Anuales, Telefono, pdf_informe, plan_stripe, dataset_vendido)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE)
                ON CONFLICT (empresa, ID_Cliente) DO UPDATE
                SET pdf_informe = EXCLUDED.pdf_informe, plan_stripe = EXCLUDED.plan_stripe
                """,
                (
                    empresa,
                    0,
                    "informe_premium",
                    "informe@spacenet.local",
                    0,
                    0,
                    "000",
                    psycopg2.Binary(pdf_bytes),
                    plan,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def guardar_dataset_vendido(empresa, df_limpio, dataset_nombre, plan=None, precio_eur=None):
    """Guarda el dataset purificado marcando dataset_vendido = True (marketplace)."""
    asegurar_columnas_premium()
    plan = plan or st.session_state.get("plan_activo") or "Pro"
    if precio_eur is None:
        precio_eur = precio_dataset_segun_plan(plan)
    conn = obtener_conexion()
    cursor = conn.cursor()
    ahora = datetime.utcnow()
    try:
        valores = [
            (
                empresa,
                int(row['ID_Cliente']),
                row['Nombre'],
                str(row['Email']),
                float(row['Edad']),
                float(row['Ingresos_Anuales']),
                str(row['Telefono']),
                True,
                dataset_nombre,
                ahora,
                plan,
                int(precio_eur),
            )
            for _, row in df_limpio.iterrows()
        ]
        query = """
            INSERT INTO clientes_purificados
                (empresa, ID_Cliente, Nombre, Email, Edad, Ingresos_Anuales, Telefono,
                 dataset_vendido, dataset_nombre, fecha_dataset, plan_stripe, precio_dataset)
            VALUES %s
            ON CONFLICT (empresa, ID_Cliente) DO UPDATE SET
                Nombre = EXCLUDED.Nombre,
                Email = EXCLUDED.Email,
                Edad = EXCLUDED.Edad,
                Ingresos_Anuales = EXCLUDED.Ingresos_Anuales,
                Telefono = EXCLUDED.Telefono,
                dataset_vendido = TRUE,
                dataset_nombre = EXCLUDED.dataset_nombre,
                fecha_dataset = EXCLUDED.fecha_dataset,
                plan_stripe = EXCLUDED.plan_stripe,
                precio_dataset = EXCLUDED.precio_dataset
        """
        execute_values(cursor, query, valores)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
    return int(precio_eur)

def listar_datasets_vendidos(empresa, marketplace=False):
    """Lista datasets con dataset_vendido=True. marketplace=True → todos los tenants."""
    asegurar_columnas_premium()
    conn = obtener_conexion()
    try:
        if marketplace or empresa == "Spacenet_Admin":
            df = pd.read_sql_query(
                """
                SELECT empresa, dataset_nombre, COUNT(*) AS registros,
                       MAX(fecha_dataset) AS fecha,
                       MAX(precio_dataset) AS precio_eur,
                       MAX(plan_stripe) AS plan_stripe
                FROM clientes_purificados
                WHERE dataset_vendido = TRUE AND dataset_nombre IS NOT NULL
                GROUP BY empresa, dataset_nombre
                ORDER BY fecha DESC NULLS LAST
                """,
                conn,
            )
        else:
            df = pd.read_sql_query(
                """
                SELECT empresa, dataset_nombre, COUNT(*) AS registros,
                       MAX(fecha_dataset) AS fecha,
                       MAX(precio_dataset) AS precio_eur,
                       MAX(plan_stripe) AS plan_stripe
                FROM clientes_purificados
                WHERE dataset_vendido = TRUE
                  AND dataset_nombre IS NOT NULL
                  AND empresa = %s
                GROUP BY empresa, dataset_nombre
                ORDER BY fecha DESC NULLS LAST
                """,
                conn,
                params=(empresa,),
            )
        if not df.empty:
            df["precio_eur"] = df.apply(
                lambda r: int(r["precio_eur"])
                if pd.notna(r["precio_eur"])
                else precio_dataset_segun_plan(r.get("plan_stripe") or "Pro"),
                axis=1,
            )
        return df
    finally:
        conn.close()

def descargar_dataset_vendido(empresa, dataset_nombre):
    conn = obtener_conexion()
    try:
        df = pd.read_sql_query(
            """
            SELECT ID_Cliente, Nombre, Email, Edad, Ingresos_Anuales, Telefono,
                   dataset_nombre, fecha_dataset
            FROM clientes_purificados
            WHERE empresa = %s AND dataset_nombre = %s AND dataset_vendido = TRUE
            ORDER BY ID_Cliente
            """,
            conn,
            params=(empresa, dataset_nombre),
        )
        return df
    finally:
        conn.close()

def obtener_fuente_ultimo_dataset():
    """Último CSV/XLSX procesado (para venta tras Checkout)."""
    df_fuente = st.session_state.get("df_sucio_ultimo")
    if df_fuente is None and st.session_state.get("analisis") is not None:
        df_fuente = st.session_state.analisis.drop(columns=["Error_IA"], errors="ignore")
    return df_fuente

def ejecutar_venta_dataset_limpio():
    """Purifica el último dataset, lo guarda con dataset_vendido=True y notifica."""
    empresa_actual = st.session_state["empresa"]
    df_fuente = obtener_fuente_ultimo_dataset()
    if df_fuente is None:
        return False, "No hay un dataset reciente. Ejecuta el Pipeline o sube un CSV primero."

    plan = st.session_state.get("plan_activo") or "Pro"
    if plan not in PLANES_PRO_O_SUPERIOR and not es_admin_tenant():
        return False, MSG_PDF_REQUIERE_PRO

    plan_precio = plan if plan in PLANES_PRO_O_SUPERIOR else "Pro"
    precio = precio_dataset_segun_plan(plan_precio)
    df_limpio, total, nulos, alertas, analisis = purificar_datos_con_ia(df_fuente.copy())
    nombre_ds = f"marketplace_{empresa_actual}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    precio = guardar_dataset_vendido(
        empresa_actual,
        df_limpio,
        nombre_ds,
        plan=plan if plan in PLANES else "Pro",
        precio_eur=precio,
    )
    st.session_state.df_procesado = df_limpio
    st.session_state.analisis = analisis
    st.session_state.metricas = (total, nulos, alertas)
    st.session_state.ultimo_dataset_vendido = nombre_ds
    try:
        enviar_aviso_dataset(
            f"📦 DATASET PUBLICADO: {nombre_ds} ({precio}€)",
            (
                f"Empresa vendedora: {empresa_actual}\n"
                f"Dataset: {nombre_ds}\n"
                f"Registros limpios: {len(df_limpio)}\n"
                f"Anomalías bloqueadas: {alertas}\n"
                f"Precio marketplace: {precio}€\n"
                f"Plan: {plan}\n"
            ),
        )
    except Exception:
        pass
    return True, (
        f"Dataset «{nombre_ds}» purificado y publicado a {precio}€ "
        f"({len(df_limpio):,} filas, {alertas:,} anomalías bloqueadas)."
    )

def iniciar_checkout_venta_dataset():
    """
    Crea la sesión Stripe Checkout (49€ / 99€) para «Vender Dataset Limpio».
    Devuelve (ok, url|None, mensaje).
    """
    if not tiene_suscripcion_pro_o_superior():
        return False, None, MSG_PDF_REQUIERE_PRO

    email = obtener_email_facturacion()
    if not email or "@" not in email:
        return False, None, "Introduce tu email de facturación en el sidebar antes de vender."

    if obtener_fuente_ultimo_dataset() is None:
        return False, None, (
            "No hay un dataset reciente para vender. "
            "Ejecuta el Pipeline de Auditoría o sube un CSV en «Dataset para mi IA» primero."
        )

    empresa = st.session_state["empresa"]
    plan = st.session_state.get("plan_activo") or "Pro"
    if plan not in PLANES_PRO_O_SUPERIOR:
        plan = "Pro"
    precio = precio_dataset_segun_plan(plan)

    try:
        session = crear_sesion_checkout_venta_dataset(email, empresa, plan)
    except Exception as e:
        return False, None, f"Error al crear Stripe Checkout: {e}"

    st.session_state.pendiente_venta_dataset = {
        "empresa": empresa,
        "plan": plan,
        "precio_eur": precio,
        "session_id": session.id,
    }
    return True, session.url, f"Abriendo Stripe Checkout · {precio}€"

def procesar_pago_exitoso(session_id):
    """Tras Checkout: suscripción, venta de dataset o compra marketplace."""
    if st.session_state.get("checkout_procesado") == session_id:
        return

    session = stripe.checkout.Session.retrieve(session_id)
    if session.get("payment_status") not in ("paid", "no_payment_required"):
        st.warning("El pago aún no está confirmado. Espera unos segundos y recarga.")
        return

    meta = session.get("metadata") or {}
    email = session.get("customer_details", {}).get("email") if session.get("customer_details") else None
    if not email and session.get("customer"):
        customer = stripe.Customer.retrieve(session["customer"])
        email = customer.get("email")
    email = email or obtener_email_facturacion() or "cliente@desconocido.com"

    # --- Venta: pago 49€/99€ → purificar + dataset_vendido=True + lista ---
    if meta.get("tipo") == "venta_dataset":
        empresa = meta.get("empresa") or session.get("client_reference_id") or st.session_state.get("empresa")
        plan = meta.get("plan") or st.session_state.get("plan_activo") or "Pro"
        precio_eur = meta.get("precio_eur") or str(precio_dataset_segun_plan(plan))
        st.session_state.email_facturacion = email
        if plan in PLANES:
            st.session_state.plan_activo = plan
            st.session_state.pro_unlocked = True
        st.session_state.force_modo = "Dataset para mi IA"
        st.session_state.mostrar_lista_tras_venta = True
        st.session_state.checkout_procesado = session_id
        ok, mensaje = ejecutar_venta_dataset_limpio()
        if ok:
            st.success(f"✅ Pago de {precio_eur}€ confirmado. {mensaje}")
        else:
            st.error(
                f"Pago de {precio_eur}€ recibido, pero no se pudo guardar el dataset: {mensaje}"
            )
        st.session_state.pendiente_venta_dataset = None
        return

    # --- Compra one-shot de un dataset ya publicado ---
    if meta.get("tipo") == "dataset":
        empresa_comprador = meta.get("empresa") or session.get("client_reference_id") or st.session_state.get("empresa")
        empresa_vendedor = meta.get("empresa_vendedor") or ""
        dataset_nombre = meta.get("dataset_nombre") or ""
        precio_eur = meta.get("precio_eur") or "?"
        st.session_state.email_facturacion = email
        marcar_dataset_comprado(empresa_vendedor, dataset_nombre)
        st.session_state.checkout_procesado = session_id
        st.session_state.dataset_compra_ok = {
            "empresa": empresa_vendedor,
            "nombre": dataset_nombre,
        }
        st.session_state.force_modo = "Dataset para mi IA"
        try:
            enviar_aviso_dataset(
                f"🛒 DATASET COMPRADO: {dataset_nombre} ({precio_eur}€)",
                (
                    f"Comprador: {empresa_comprador} ({email})\n"
                    f"Vendedor: {empresa_vendedor}\n"
                    f"Dataset: {dataset_nombre}\n"
                    f"Precio: {precio_eur}€\n"
                ),
            )
        except Exception:
            pass
        st.success(
            f"✅ Dataset «{dataset_nombre}» comprado ({precio_eur}€). "
            "Ya puedes descargarlo en Datasets para mi IA."
        )
        return

    empresa = (
        meta.get("empresa")
        or session.get("client_reference_id")
        or st.session_state.get("empresa")
    )
    plan = meta.get("plan") or "Pro"
    trial = meta.get("trial") == "1"
    nombre = empresa or "Cliente"

    st.session_state.pro_unlocked = True
    st.session_state.email_facturacion = email
    st.session_state.plan_activo = plan

    # PDF premium solo con Pro o Business
    if plan in PLANES_PRO_O_SUPERIOR:
        total, nulos, alertas = obtener_metricas_para_pdf(empresa)
        pdf_bytes = generar_reporte_pdf(total, nulos, alertas, empresa)
        st.session_state.pdf_generado = pdf_bytes

        try:
            guardar_pdf_en_db(empresa, pdf_bytes, plan)
        except Exception as e:
            st.error(f"Pago OK, pero no se pudo guardar el PDF en la base de datos: {e}")

        try:
            enviar_aviso_pago(nombre, email, empresa, plan, trial=trial)
        except Exception:
            pass

        st.session_state.checkout_procesado = session_id
        st.success(f"✅ Suscripción {plan} activada. PDF premium generado y guardado.")
    else:
        st.session_state.pdf_generado = None
        try:
            enviar_aviso_pago(nombre, email, empresa, plan, trial=trial)
        except Exception:
            pass
        st.session_state.checkout_procesado = session_id
        st.success(f"✅ Suscripción {plan} activada.")
        st.info(
            "📄 PDF premium requiere suscripción activa (Pro o Business). "
            "Actualiza tu plan para generar y descargar el informe."
        )

# ==========================================
# DASHBOARD B2B (MULTI-TENANT)
# ==========================================
if "pro_unlocked" not in st.session_state:
    st.session_state.pro_unlocked = False
if "pdf_generado" not in st.session_state:
    st.session_state.pdf_generado = None
if "email_facturacion" not in st.session_state:
    st.session_state.email_facturacion = ""
if "plan_activo" not in st.session_state:
    st.session_state.plan_activo = None
if "datasets_comprados" not in st.session_state:
    st.session_state.datasets_comprados = set()
if "accion_vender_dataset" not in st.session_state:
    st.session_state.accion_vender_dataset = False
if "df_sucio_ultimo" not in st.session_state:
    st.session_state.df_sucio_ultimo = None

# Retorno desde Stripe Checkout
params = st.query_params
if params.get("checkout") == "success" and params.get("session_id"):
    try:
        procesar_pago_exitoso(params.get("session_id"))
    except Exception as e:
        st.error(f"Error al procesar el pago: {e}")
elif params.get("checkout") == "cancel":
    st.warning("Checkout cancelado. Puedes reintentar cuando quieras.")

st.sidebar.title(f"Espacio: {st.session_state['empresa']}")
st.sidebar.markdown("**Usuario Operativo**")
if st.sidebar.button("Cerrar Sesión"):
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("💳 Suscripción Stripe")
email_sidebar = st.sidebar.text_input(
    "Email de facturación",
    value=st.session_state.get("email_facturacion", ""),
    key="email_facturacion_sidebar",
    help="Email usado en Stripe Checkout",
)
if email_sidebar:
    st.session_state.email_facturacion = email_sidebar.strip()

if st.sidebar.button("Ver mis suscripciones"):
    email = obtener_email_facturacion()
    if not email:
        st.sidebar.error("Introduce tu email de facturación.")
    else:
        try:
            portal = crear_sesion_portal(email, st.session_state["empresa"])
            st.sidebar.link_button("Abrir Customer Portal", portal.url, type="primary")
            st.sidebar.info("Se abrirá el portal de Stripe para gestionar facturas y métodos de pago.")
        except Exception as e:
            st.sidebar.error(f"No se pudo abrir el portal: {e}")

if st.sidebar.button("Cancelar suscripción"):
    email = obtener_email_facturacion()
    try:
        ok, mensaje = cancelar_suscripcion_stripe(email)
        if ok:
            st.session_state.pro_unlocked = False
            st.session_state.plan_activo = None
            st.session_state.pdf_generado = None
            st.sidebar.success(mensaje)
        else:
            st.sidebar.error(mensaje)
    except Exception as e:
        st.sidebar.error(f"Error al cancelar: {e}")

st.sidebar.markdown("---")

# MÓDULOS DE ADMINISTRADOR
if st.session_state["empresa"] == "Spacenet_Admin":
    st.sidebar.subheader("🛠️ Zona Root (Solo Admin)")

    if st.sidebar.button("1. Actualizar Arquitectura SQL"):
        with st.spinner("Modificando base de datos..."):
            conn = obtener_conexion()
            cursor = conn.cursor()
            cursor.execute(
                "ALTER TABLE clientes_purificados ADD COLUMN IF NOT EXISTS empresa VARCHAR(100) DEFAULT 'Desconocida'"
            )
            conn.commit()

            try:
                cursor.execute("ALTER TABLE clientes_purificados DROP CONSTRAINT clientes_purificados_pkey")
                conn.commit()
            except Exception:
                conn.rollback()

            try:
                cursor.execute(
                    "ALTER TABLE clientes_purificados ADD CONSTRAINT clientes_purificados_pkey PRIMARY KEY (empresa, ID_Cliente)"
                )
                conn.commit()
            except Exception:
                conn.rollback()

            cursor.close()
            conn.close()
            try:
                asegurar_columnas_premium()
            except Exception as e:
                st.sidebar.error(f"Columnas premium: {e}")
        st.sidebar.success("Base de datos adaptada a Multi-Tenant!")

    if st.sidebar.button("2. Vaciar Servidor Global"):
        conn = obtener_conexion()
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE clientes_purificados RESTART IDENTITY")
        conn.commit()
        cursor.close()
        conn.close()
        st.sidebar.warning("Todas las bases de datos eliminadas.")
    st.sidebar.markdown("---")

modo = st.sidebar.radio(
    "Módulos",
    ["Pipeline de Auditoría", "Base de Datos SQL", "Dataset para mi IA"],
    key="modo_radio",
)

# Data Flywheel — botones bajo el acceso a Dataset para mi IA
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Data Flywheel")
if st.sidebar.button("Crear dataset para mi IA", key="sidebar_crear_dataset"):
    st.session_state.force_modo = "Dataset para mi IA"
    st.session_state.accion_vender_dataset = False
    st.session_state.checkout_venta_url = None
    st.rerun()

if tiene_suscripcion_pro_o_superior():
    plan_sb = st.session_state.get("plan_activo") or "Pro"
    precio_sb = precio_dataset_segun_plan(
        plan_sb if plan_sb in PLANES_PRO_O_SUPERIOR else "Pro"
    )
    if st.sidebar.button(
        f"Vender Dataset Limpio para mi IA ({precio_sb}€)",
        key="sidebar_vender_dataset",
    ):
        ok, url, mensaje = iniciar_checkout_venta_dataset()
        if ok and url:
            st.session_state.checkout_venta_url = url
            st.session_state.checkout_venta_abierto = False
            st.session_state.force_modo = "Dataset para mi IA"
            st.session_state.accion_vender_dataset = False
            st.rerun()
        else:
            st.sidebar.error(mensaje)
else:
    st.sidebar.caption("Venta de datasets: requiere Pro o superior.")

if st.session_state.get("checkout_venta_url"):
    st.sidebar.link_button(
        "Abrir Stripe Checkout →",
        st.session_state.checkout_venta_url,
        type="primary",
    )
    st.sidebar.caption("Completa el pago para publicar el dataset limpio.")

if st.session_state.get("force_modo"):
    modo = st.session_state.pop("force_modo")

st.title("Spacenet Data Intelligence")

# Si hay URL de Checkout pendiente (venta), abrir Stripe una sola vez
if st.session_state.get("checkout_venta_url"):
    url_venta = st.session_state.checkout_venta_url
    st.info("💳 Checkout de venta de dataset listo. Completa el pago en Stripe.")
    if not st.session_state.get("checkout_venta_abierto"):
        st.session_state.checkout_venta_abierto = True
        abrir_url_stripe(url_venta, etiqueta="Pagar y publicar dataset en Stripe →")
    else:
        st.link_button(
            "Pagar y publicar dataset en Stripe →",
            url_venta,
            type="primary",
        )

def render_marketplace_datasets(contexto="marketplace", solo_propios=False):
    """Lista datasets limpios con precio y Checkout Stripe para comprar."""
    st.markdown("### Datasets para mi IA")
    st.caption(
        "Datasets purificados listos para entrenar tu propia IA. "
        f"Precio: {PRECIO_DATASET_PRO_EUR}€ (Pro) · {PRECIO_DATASET_BUSINESS_EUR}€ (Business)."
    )

    empresa_actual = st.session_state["empresa"]
    try:
        df_datasets = listar_datasets_vendidos(
            empresa_actual,
            marketplace=not solo_propios,
        )
    except Exception as e:
        st.error(
            "No se pudieron listar los datasets. "
            f"Ejecuta «Actualizar Arquitectura SQL» como admin. Detalle: {e}"
        )
        return

    if df_datasets.empty:
        st.caption("Aún no hay datasets limpios en el marketplace.")
        return

    for _, row in df_datasets.iterrows():
        empresa_ds = row["empresa"]
        nombre_ds = row["dataset_nombre"]
        registros = int(row["registros"])
        fecha = row["fecha"]
        precio = int(row["precio_eur"]) if pd.notna(row.get("precio_eur")) else PRECIO_DATASET_PRO_EUR
        es_propio = empresa_ds == empresa_actual or es_admin_tenant()
        comprado = dataset_ya_comprado(empresa_ds, nombre_ds)
        puede_descargar = es_propio or comprado

        c1, c2 = st.columns([3, 1])
        with c1:
            st.write(
                f"**{nombre_ds}** · {empresa_ds} · {registros:,} registros · "
                f"{fecha} · **{precio}€**"
            )
        with c2:
            if puede_descargar:
                try:
                    df_dl = descargar_dataset_vendido(empresa_ds, nombre_ds)
                    csv_bytes = df_dl.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Descargar",
                        data=csv_bytes,
                        file_name=f"{nombre_ds}.csv",
                        mime="text/csv",
                        key=f"dl_{contexto}_{empresa_ds}_{nombre_ds}",
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                if st.button(
                    f"Comprar {precio}€",
                    key=f"buy_{contexto}_{empresa_ds}_{nombre_ds}",
                ):
                    email = obtener_email_facturacion()
                    if not email or "@" not in email:
                        st.error("Introduce tu email de facturación en el sidebar.")
                    else:
                        try:
                            session = crear_sesion_checkout_dataset(
                                email,
                                empresa_actual,
                                empresa_ds,
                                nombre_ds,
                                precio,
                            )
                            abrir_url_stripe(
                                session.url,
                                etiqueta=f"Pagar {precio}€ en Stripe →",
                            )
                        except Exception as e:
                            st.error(f"Error al crear Checkout: {e}")

def render_checkout_planes(contexto="paywall"):
    """UI de Stripe Checkout: Pagar X€ / Probar gratis (multi-plan)."""
    st.markdown("### Planes Spacenet AI")
    st.caption("Starter 79€ · Pro 199€ · Business 499€ / mes")

    email = st.text_input(
        "Email de facturación para Stripe",
        value=st.session_state.get("email_facturacion", ""),
        key=f"email_checkout_{contexto}",
    )
    if email:
        st.session_state.email_facturacion = email.strip()

    cols = st.columns(3)
    for col, plan in zip(cols, ["Starter", "Pro", "Business"]):
        info = PLANES[plan]
        with col:
            st.markdown(f"**{plan} — {info['precio_eur']}€/mes**")
            st.caption(info["descripcion"])
            if st.button(f"Pagar {info['precio_eur']}€", key=f"pagar_{plan}_{contexto}"):
                if not obtener_email_facturacion() or "@" not in obtener_email_facturacion():
                    st.error("Introduce un email de facturación válido.")
                else:
                    try:
                        session = crear_sesion_checkout(
                            obtener_email_facturacion(),
                            st.session_state["empresa"],
                            plan=plan,
                            trial=False,
                        )
                        st.link_button("Ir a Stripe Checkout →", session.url, type="primary")
                    except Exception as e:
                        st.error(f"Error al crear Checkout: {e}")

            if plan == "Pro":
                if st.button("Probar gratis", key=f"trial_{plan}_{contexto}"):
                    if not obtener_email_facturacion() or "@" not in obtener_email_facturacion():
                        st.error("Introduce un email de facturación válido.")
                    else:
                        try:
                            session = crear_sesion_checkout(
                                obtener_email_facturacion(),
                                st.session_state["empresa"],
                                plan=plan,
                                trial=True,
                            )
                            st.link_button("Activar prueba de 7 días →", session.url, type="primary")
                        except Exception as e:
                            st.error(f"Error al crear prueba gratuita: {e}")

    with st.expander("🔑 ¿Ya tienes licencia? Verifica tu acceso"):
        email_pago = st.text_input(
            "Email de facturación registrado en Stripe",
            key=f"verificar_{contexto}",
        )
        if st.button("Verificar Suscripción", key=f"btn_verificar_{contexto}"):
            with st.spinner("Conectando con Stripe..."):
                plan = sincronizar_plan_desde_stripe(email_pago)
                if plan or verificar_suscripcion_activa(email_pago):
                    st.session_state.pro_unlocked = True
                    st.session_state.email_facturacion = email_pago.strip()
                    if plan:
                        st.session_state.plan_activo = plan
                    st.success(
                        f"✅ Licencia validada ({st.session_state.get('plan_activo') or 'activa'}). ¡Acceso concedido!"
                    )
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(
                        "❌ No se ha encontrado una suscripción activa para este email. "
                        "Si acabas de suscribirte, espera unos segundos."
                    )

if modo == "Pipeline de Auditoría":
    st.subheader("Ingesta y Procesamiento Neuronal")
    archivo = st.file_uploader("Cargar dataset", type=["csv", "xlsx"])

    if 'df_procesado' not in st.session_state:
        st.session_state.df_procesado = None
        st.session_state.analisis = None
        st.session_state.metricas = None

    if archivo and st.button("Ejecutar Análisis"):
        with st.spinner("Procesando red neuronal..."):
            df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
            empresa_actual = st.session_state["empresa"]
            df = adaptar_columnas_tenant(df, empresa_actual)
            df_limpio, total, nulos, alertas, analisis = purificar_datos_con_ia(df)

            cursor = None
            conn = None
            try:
                conn = obtener_conexion()
                cursor = conn.cursor()
                valores = [
                    (
                        empresa_actual,
                        int(row['ID_Cliente']),
                        row['Nombre'],
                        str(row['Email']),
                        float(row['Edad']),
                        float(row['Ingresos_Anuales']),
                        str(row['Telefono']),
                    )
                    for _, row in df_limpio.iterrows()
                ]
                query = (
                    "INSERT INTO clientes_purificados "
                    "(empresa, ID_Cliente, Nombre, Email, Edad, Ingresos_Anuales, Telefono) "
                    "VALUES %s ON CONFLICT (empresa, ID_Cliente) DO NOTHING"
                )
                execute_values(cursor, query, valores)
                conn.commit()
            except Exception as e:
                if conn:
                    conn.rollback()
                st.error(f"Error al guardar en DB: {e}")
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()

            st.session_state.df_procesado = df_limpio
            st.session_state.analisis = analisis
            st.session_state.metricas = (total, nulos, alertas)
            st.session_state.df_sucio_ultimo = df.copy()
            st.session_state.pdf_generado = None

    if st.session_state.df_procesado is not None:
        df_limpio = st.session_state.df_procesado
        analisis = st.session_state.analisis
        total, nulos, alertas = st.session_state.metricas

        col1, col2, col3 = st.columns(3)
        col1.metric("Registros Auditados", f"{total:,}")
        col2.metric("Nulos Corregidos", f"{nulos:,}")
        col3.metric("Anomalías Bloqueadas", f"{alertas:,}")

        st.write("Gráfico de dispersión de error (Autoencoder):")
        st.line_chart(analisis['Error_IA'].head(100))

        tab1, tab2 = st.tabs(["Datos Purificados", "Sala de Cuarentena"])
        with tab1:
            st.dataframe(df_limpio.drop(columns=['Email_Roto', 'Nombre_Falso'], errors='ignore'), width='stretch')
        with tab2:
            st.dataframe(analisis[analisis['Error_IA'] > 0.05], width='stretch')
            st.markdown("---")
            render_marketplace_datasets(contexto="cuarentena")

        with st.expander("📥 Obtener Informe de Auditoría Completo", expanded=True):
            st.info(MSG_PDF_PREMIUM)

            if not tiene_suscripcion_pro_o_superior():
                # Bloquea generación y cualquier PDF residual en sesión
                st.session_state.pdf_generado = None
                st.warning(MSG_PDF_REQUIERE_PRO)
                render_checkout_planes(contexto="pdf")
            else:
                st.write("Introduce tus datos para generar y descargar el informe oficial.")
                with st.form("form_captacion"):
                    nombre_cliente = st.text_input("Nombre de la Empresa / Contacto")
                    email_cliente = st.text_input("Email Corporativo")
                    st.caption(MSG_PDF_PREMIUM)
                    submit_button = st.form_submit_button("Generar PDF")

                    if submit_button:
                        if not tiene_suscripcion_pro_o_superior():
                            st.session_state.pdf_generado = None
                            st.warning(MSG_PDF_REQUIERE_PRO)
                        elif email_cliente and "@" in email_cliente:
                            empresa_actual = st.session_state["empresa"]
                            st.session_state.pdf_generado = generar_reporte_pdf(
                                total, nulos, alertas, empresa_actual
                            )
                            try:
                                enviar_aviso_venta(nombre_cliente, email_cliente, empresa_actual)
                            except Exception:
                                pass
                            st.success(f"Informe listo para {nombre_cliente}. ¡Ya puedes descargarlo!")
                        else:
                            st.error("Por favor, introduce un email corporativo válido.")

                if st.session_state.pdf_generado and tiene_suscripcion_pro_o_superior():
                    st.download_button(
                        label="⬇️ DESCARGAR PDF AHORA",
                        data=st.session_state.pdf_generado,
                        file_name=f"Informe_Auditoria_{st.session_state['empresa']}.pdf",
                        mime="application/pdf",
                        key="dl_pdf_pipeline",
                    )
                elif st.session_state.pdf_generado:
                    st.session_state.pdf_generado = None
    else:
        # Sin análisis previo: marketplace visible en el dashboard del Pipeline
        st.markdown("---")
        render_marketplace_datasets(contexto="pipeline_sin_datos")

elif modo == "Base de Datos SQL":
    st.subheader(f"Registros Aislados: {st.session_state['empresa']}")
    conn = obtener_conexion()

    if st.session_state["empresa"] == "Spacenet_Admin":
        df_sql = pd.read_sql_query("SELECT * FROM clientes_purificados", conn)
    else:
        df_sql = pd.read_sql_query(
            "SELECT * FROM clientes_purificados WHERE empresa = %s",
            conn,
            params=(st.session_state['empresa'],),
        )
    conn.close()

    # No mostrar el BYTEA del PDF en la tabla
    cols_mostrar = [c for c in df_sql.columns if c != "pdf_informe"]
    st.dataframe(df_sql[cols_mostrar] if cols_mostrar else df_sql, width='stretch')

    # --- MURO DE PAGO (STRIPE CHECKOUT) ---
    if not df_sql.empty:
        es_admin = (st.session_state.get("empresa") == "Spacenet_Admin")

        if not es_admin and not st.session_state.pro_unlocked:
            st.warning("⚠️ Los datos purificados están bloqueados. Se requiere acceso de nivel Corporativo.")
            st.link_button(
                "💼 SOLICITAR ACCESO CORPORATIVO",
                "mailto:joelrodriguezcr10@gmail.com?subject=Consulta Acceso Spacenet AI",
            )
            render_checkout_planes(contexto="sql")
        else:
            if es_admin:
                st.success("👑 Modo Administrador: Muro de pago ignorado. Acceso total habilitado.")
            else:
                plan_txt = st.session_state.get("plan_activo") or "PRO"
                st.success(f"✅ Licencia {plan_txt} activada. Descarga habilitada.")

            csv = df_sql.drop(columns=["pdf_informe"], errors="ignore").to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ EXPORTAR DATASET PURIFICADO (CSV)",
                data=csv,
                file_name=f"dataset_limpio_{st.session_state['empresa']}.csv",
                mime="text/csv",
            )

            if st.session_state.pdf_generado and tiene_suscripcion_pro_o_superior():
                st.info(MSG_PDF_PREMIUM)
                st.download_button(
                    label="⬇️ DESCARGAR PDF PREMIUM",
                    data=st.session_state.pdf_generado,
                    file_name=f"Informe_Premium_{st.session_state['empresa']}.pdf",
                    mime="application/pdf",
                    key="dl_pdf_sql",
                )
            elif st.session_state.pdf_generado and not tiene_suscripcion_pro_o_superior():
                st.session_state.pdf_generado = None
                st.warning(MSG_PDF_REQUIERE_PRO)

elif modo == "Dataset para mi IA":
    st.subheader("Dataset para mi IA · Data Flywheel")
    st.write(
        "Crea, publica y compra datasets purificados con el autoencoder. "
        "Los datasets listos se marcan con `dataset_vendido = True` en `clientes_purificados`."
    )

    if not es_admin_tenant() and not st.session_state.pro_unlocked:
        st.info("Este módulo requiere suscripción activa. Elige un plan para continuar.")
        render_checkout_planes(contexto="dataset")
    else:
        if st.session_state.get("mostrar_lista_tras_venta"):
            st.session_state.mostrar_lista_tras_venta = False
            st.session_state.checkout_venta_url = None
            st.session_state.checkout_venta_abierto = False

        st.markdown("#### Crear dataset para mi IA")
        archivo_ia = st.file_uploader(
            "Cargar dataset sucio para purificar",
            type=["csv", "xlsx"],
            key="uploader_dataset_ia",
        )
        nombre_dataset = st.text_input(
            "Nombre del dataset",
            value=f"dataset_{st.session_state['empresa']}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            crear_clicked = st.button(
                "Crear dataset para mi IA",
                type="primary",
                key="btn_crear_dataset_mod",
            )
        with col_b:
            vender_clicked = False
            if tiene_suscripcion_pro_o_superior():
                plan_mod = st.session_state.get("plan_activo") or "Pro"
                precio_mod = precio_dataset_segun_plan(
                    plan_mod if plan_mod in PLANES_PRO_O_SUPERIOR else "Pro"
                )
                vender_clicked = st.button(
                    f"Vender Dataset Limpio para mi IA ({precio_mod}€)",
                    key="btn_vender_dataset_mod",
                )
            else:
                st.caption("Venta: requiere Pro o superior")

        if crear_clicked:
            if not archivo_ia:
                st.error("Sube un archivo CSV o XLSX primero.")
            elif not nombre_dataset.strip():
                st.error("Indica un nombre para el dataset.")
            else:
                with st.spinner("Purificando con autoencoder..."):
                    df = (
                        pd.read_csv(archivo_ia)
                        if archivo_ia.name.endswith('.csv')
                        else pd.read_excel(archivo_ia)
                    )
                    empresa_actual = st.session_state["empresa"]
                    df = adaptar_columnas_tenant(df, empresa_actual)
                    st.session_state.df_sucio_ultimo = df.copy()
                    df_limpio, total, nulos, alertas, analisis = purificar_datos_con_ia(df)
                    try:
                        st.session_state.metricas = (total, nulos, alertas)
                        st.session_state.df_procesado = df_limpio
                        st.session_state.analisis = analisis
                        st.success(
                            f"Dataset «{nombre_dataset.strip()}» purificado "
                            f"({len(df_limpio):,} filas, {alertas:,} anomalías bloqueadas). "
                            "Usa «Vender Dataset Limpio» para publicarlo vía Stripe Checkout."
                        )
                    except Exception as e:
                        st.error(f"Error al procesar el dataset: {e}")

        if vender_clicked:
            if not tiene_suscripcion_pro_o_superior():
                st.warning(MSG_PDF_REQUIERE_PRO)
            else:
                if archivo_ia:
                    df = (
                        pd.read_csv(archivo_ia)
                        if archivo_ia.name.endswith('.csv')
                        else pd.read_excel(archivo_ia)
                    )
                    df = adaptar_columnas_tenant(df, st.session_state["empresa"])
                    st.session_state.df_sucio_ultimo = df.copy()
                with st.spinner("Creando Stripe Checkout..."):
                    ok, url, mensaje = iniciar_checkout_venta_dataset()
                if ok and url:
                    st.session_state.checkout_venta_url = url
                    st.session_state.checkout_venta_abierto = True
                    st.success(mensaje)
                    abrir_url_stripe(url, etiqueta="Pagar y publicar dataset en Stripe →")
                else:
                    st.error(mensaje)

        st.markdown("---")
        render_marketplace_datasets(contexto="modulo_dataset")
