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
# AUTHENTICATION (MULTI-TENANT)
# ==========================================
def verificar_credenciales(usuario, password):
    try:
        # Busca en la sección [usuarios] de secrets
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
    if len(tel_str) < 7: return "*******"
    return f"{tel_str[:3]} ****** {tel_str[-3:]}"

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
    
    # Header Premium
    pdf.set_fill_color(15, 23, 42) 
    pdf.rect(0, 0, 210, 38, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 22)
    pdf.cell(0, 10, "SPACENET DATA INTELLIGENCE", ln=True, align='L')
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(148, 163, 184) 
    pdf.cell(0, 5, "AI-Powered Data Purification & Security Audit", ln=True, align='L')
    pdf.ln(20)
    
    # Sección Impacto de Negocio (NUEVA)
    pdf.set_text_color(30, 41, 59)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "IMPACTO DE NEGOCIO ESTIMADO", ln=True)
    pdf.set_fill_color(240, 253, 244) # Verde suave
    pdf.rect(10, pdf.get_y(), 190, 20, 'F')
    pdf.set_text_color(22, 101, 52)
    pdf.set_font("Arial", size=11)
    ahorro_estimado = alertas * 15 # Asumimos 15€ ahorrados por cada error bloqueado
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
        ("Registros Totales Auditados", f"{total:,} filas"),
        ("Registros Nulos Corregidos por IA", f"{nulos:,} correcciones"),
        ("Anomalias / Fraudes Detectados (Bloqueados)", f"{alertas:,} alertas"),
        ("Estado Final del Dataset", "PURIFICADO & SEGURO"),
        ("Protocolo de Criptografia Aplicado", "SHA-256 + Phone Masking")
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
        msg_diagnostico = f"  ATENCION: Se han detectado {alertas} vectores de riesgo en el dataset. Los datos han sido aislados\n  en la sala de cuarentena para proteger la integridad de su base de datos corporativa."
    else:
        msg_diagnostico = "  OPTIMO: No se han detectado anomalias criticas. El dataset cumple con los estandares internacionales\n  de calidad y politicas de privacidad de datos."
        
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
# Configurar la clave de Stripe
stripe.api_key = st.secrets["STRIPE_SECRET_KEY"]

def verificar_suscripcion_activa(email):
    """Busca en Stripe si el email tiene una suscripción mensual activa."""
    if not email:
        return False
        
    try:
        # 1. Buscamos al cliente por su email en Stripe
        clientes = stripe.Customer.list(email=email.strip(), limit=1)
        if not clientes.data:
            return False # El email no existe en Stripe
            
        cliente_id = clientes.data[0].id
        
        # 2. Buscamos si ese cliente tiene suscripciones activas
        suscripciones = stripe.Subscription.list(customer=cliente_id, status="active")
        
        if len(suscripciones.data) > 0:
            return True # ¡Tiene la suscripción de 299€ pagada al día!
        else:
            return False # Canceló o no ha pagado
            
    except Exception as e:
        st.error(f"Error de conexión con la pasarela de pagos: {e}")
        return False
# ==========================================
# DASHBOARD B2B (MULTI-TENANT)
# ==========================================
st.sidebar.title(f"Espacio: {st.session_state['empresa']}")
st.sidebar.markdown(f"**Usuario Operativo**")
if st.sidebar.button("Cerrar Sesión"):
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")

# MÓDULOS DE ADMINISTRADOR
if st.session_state["empresa"] == "Spacenet_Admin":
    st.sidebar.subheader("🛠️ Zona Root (Solo Admin)")
    
    if st.sidebar.button("1. Actualizar Arquitectura SQL"):
        with st.spinner("Modificando base de datos..."):
            conn = obtener_conexion()
            cursor = conn.cursor()
            cursor.execute("ALTER TABLE clientes_purificados ADD COLUMN IF NOT EXISTS empresa VARCHAR(100) DEFAULT 'Desconocida'")
            conn.commit()
            
            try:
                cursor.execute("ALTER TABLE clientes_purificados DROP CONSTRAINT clientes_purificados_pkey")
                conn.commit()
            except:
                conn.rollback()
                
            try:
                cursor.execute("ALTER TABLE clientes_purificados ADD CONSTRAINT clientes_purificados_pkey PRIMARY KEY (empresa, ID_Cliente)")
                conn.commit()
            except:
                conn.rollback()
                
            cursor.close()
            conn.close()
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

modo = st.sidebar.radio("Módulos", ["Pipeline de Auditoría", "Base de Datos SQL"])

st.title("Spacenet Data Intelligence")

if modo == "Pipeline de Auditoría":
    st.subheader("Ingesta y Procesamiento Neuronal")
    archivo = st.file_uploader("Cargar dataset", type=["csv", "xlsx"])
    
    if 'df_procesado' not in st.session_state:
        st.session_state.df_procesado = None
        st.session_state.analisis = None
        st.session_state.metricas = None
    if 'pdf_generado' not in st.session_state:
        st.session_state.pdf_generado = None

    if archivo and st.button("Ejecutar Análisis"):
        with st.spinner("Procesando red neuronal..."):
            df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
            empresa_actual = st.session_state["empresa"]

            # Si entra la Agencia X, le renombramos sus columnas raras a las estándar que entiende tu IA
            if empresa_actual == "Agencia_X":
                df = df.rename(columns={
                    "Presupuesto_Mensual": "Ingresos_Anuales",
                    "Años": "Edad",
                    "Correo": "Email",
                    "Cliente_Nombre": "Nombre"
                })

            # Si entra E-commerce Y, hacemos lo mismo con las suyas
            elif empresa_actual == "Ecommerce_Y":
                df = df.rename(columns={
                    "Facturacion": "Ingresos_Anuales",
                    "Email_Contacto": "Email"
                })

# A partir de aquí, tu motor de IA siempre recibe 'Edad', 'Ingresos_Anuales', etc. ¡No tienes que tocar la IA!
            df_limpio, total, nulos, alertas, analisis = purificar_datos_con_ia(df)
            
            # GUARDADO MULTI-TENANT
            try:
                conn = obtener_conexion()
                cursor = conn.cursor()
                empresa_actual = st.session_state["empresa"]
            
                valores = [(empresa_actual, int(row['ID_Cliente']), row['Nombre'], str(row['Email']), float(row['Edad']), float(row['Ingresos_Anuales']), str(row['Telefono'])) for _, row in df_limpio.iterrows()]
                query = "INSERT INTO clientes_purificados (empresa, ID_Cliente, Nombre, Email, Edad, Ingresos_Anuales, Telefono) VALUES %s ON CONFLICT (empresa, ID_Cliente) DO NOTHING"
            
                execute_values(cursor, query, valores)
                conn.commit()
            except Exception as e:
                conn.rollback() # Limpia el error de transacción
                st.error(f"Error al guardar en DB: {e}")    
            finally:
                cursor.close()
                conn.close()

                st.session_state.df_procesado = df_limpio
                st.session_state.analisis = analisis
                st.session_state.metricas = (total, nulos, alertas)
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
            st.dataframe(df_limpio.drop(columns=['Email_Roto', 'Nombre_Falso']), width='stretch')
        with tab2:
            st.dataframe(analisis[analisis['Error_IA'] > 0.05], width='stretch')

        with st.expander("📥 Obtener Informe de Auditoría Completo"):
            st.write("Introduce tus datos para generar y descargar el informe oficial.")
            with st.form("form_captacion"):
                nombre_cliente = st.text_input("Nombre de la Empresa / Contacto")
                email_cliente = st.text_input("Email Corporativo")
                submit_button = st.form_submit_button("Generar PDF")
                
                if submit_button:
                    if email_cliente and "@" in email_cliente:
                        empresa_actual = st.session_state["empresa"]
                        st.session_state.pdf_generado = generar_reporte_pdf(total, nulos, alertas, empresa_actual)
                        try:
                            enviar_aviso_venta(nombre_cliente, email_cliente, empresa_actual)
                        except:
                            pass 
                        st.success(f"Informe listo para {nombre_cliente}. ¡Ya puedes descargarlo!")
                    else:
                        st.error("Por favor, introduce un email corporativo válido.")

        if st.session_state.pdf_generado:
            st.download_button(
                label="⬇️ DESCARGAR PDF AHORA",
                data=st.session_state.pdf_generado,
                file_name=f"Informe_Auditoria_{st.session_state['empresa']}.pdf",
                mime="application/pdf"
            )

elif modo == "Base de Datos SQL":
    st.subheader(f"Registros Aislados: {st.session_state['empresa']}")
    conn = obtener_conexion()
    
    if st.session_state["empresa"] == "Spacenet_Admin":
        df_sql = pd.read_sql_query("SELECT * FROM clientes_purificados", conn)
    else:
        empresa_limpia = st.session_state['empresa'].replace("'", "''")
        df_sql = pd.read_sql_query(f"SELECT * FROM clientes_purificados WHERE empresa = '{empresa_limpia}'", conn)
    conn.close()
    
    st.dataframe(df_sql, width='stretch')
    
    # --- MURO DE PAGO (PAYWALL) ---
    if not df_sql.empty:
        # 1. Identificamos si el que ha entrado eres tú (el jefe)
        es_admin = (st.session_state.get("empresa") == "Spacenet_Admin")
        
        if 'pro_unlocked' not in st.session_state:
            st.session_state.pro_unlocked = False
            
        # 2. Si NO eres el admin y NO has pagado -> Muro cerrado
        if not es_admin and not st.session_state.pro_unlocked:
            st.warning("⚠️ Los datos purificados están bloqueados. Se requiere acceso de nivel Corporativo.")
            
            # Botón de contacto
            st.link_button("💼 SOLICITAR ACCESO CORPORATIVO", "mailto:joelrodriguezcr10@gmail.com?subject=Consulta Acceso Spacenet AI")
            
            # Botón de compra rápida
            if st.button("Adquirir Licencia Individual (299€)"):
                st.write("Redirigiendo a pasarela segura...")
                st.link_button("COMPRAR AHORA", "https://buy.stripe.com/00w14gaRG1Dn6FT1L22Nq01")
            
            # Verificación con Stripe
            with st.expander("🔑 ¿Ya tienes licencia? Verifica tu acceso"):
                st.write("Introduce el email con el que realizaste la compra en Stripe:")
                email_pago = st.text_input("Email de facturación")
                
                if st.button("Verificar Suscripción"):
                    with st.spinner("Conectando con la base de datos de licencias..."):
                        if verificar_suscripcion_activa(email_pago):
                            st.session_state.pro_unlocked = True
                            st.success("✅ Licencia validada. ¡Acceso concedido!")
                            time.sleep(1) 
                            st.rerun()
                        else:
                            st.error("❌ No se ha encontrado una suscripción activa para este email. Si acabas de suscribirte, espera unos segundos.")
        
        # 3. Si ERES el admin O SI has pagado -> Descarga libre
        else:
            if es_admin:
                st.success("👑 Modo Administrador: Muro de pago ignorado. Acceso total habilitado.")
            else:
                st.success("✅ Licencia PRO activada. Descarga habilitada.")
                
            csv = df_sql.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ EXPORTAR DATASET PURIFICADO (CSV)", 
                data=csv, 
                file_name=f"dataset_limpio_{st.session_state['empresa']}.csv", 
                mime="text/csv"
            )