##############
# En este módulo se definen funciones personales que pueden ser útiles.
#
# Ejemplo: from personalizar import MF
#              MF('r')
#                (antes de plt.show())
#
##############
#
#%% Función «MF» (Mover Figura)
#
import matplotlib.pyplot as plt

def MF(posicion='ur'):
  '''
  MF: «Mover Figura».

  Función basada en la función «move_figure» de la web
    https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib

  Mueve y redimensiona una ventana a un conjunto de posiciones estándar en la
  pantalla. Las posibles posiciones son:
     'u':  arriba
     'b':  abajo
     'l':  izquierda
     'r':  derecha
     'c':  centrado
     'C':  Centrado grande
     'ul': arriba izq.
     'ur': arriba dcha.
     'bl': abajo izq.
     'br': abajo dcha.
  Oviedo, 05-abril-2023
  '''
  mgr = plt.get_current_fig_manager()
  mgr.full_screen_toggle()
  py = mgr.canvas.height()
  px = mgr.canvas.width()
  mgr.full_screen_toggle()

  d = 80
  if posicion == "u":
      mgr.window.setGeometry(d, 3*d, px - 2*d, py//2 - 4*d)
  elif posicion == 'b':
      mgr.window.setGeometry(d, py//2 + 2*d, px - 2*d, py//2 - 4*d)
  elif posicion == 'l':
      mgr.window.setGeometry(d, 3*d, px//2 - 2*d, py - 4*d)
  elif posicion == 'r':
      mgr.window.setGeometry(px//2 + d, 3*d, px//2 - 2*d, py - 4*d)
  elif posicion == 'c':
      mgr.window.setGeometry(px//8, py//8, 6*px//8, 6*py//8)
  elif posicion == 'C':
      mgr.window.setGeometry(d//10, d, px-d//5, py-2*d)
  elif posicion == 'ul':
      mgr.window.setGeometry(d, 3*d, px//2 - 2*d, py//2 - 4*d)
  elif posicion == 'ur':
      mgr.window.setGeometry(px//2 + d, 3*d, px//2 - 2*d, py//2 - 4*d)
  elif posicion == 'bl':
      mgr.window.setGeometry(d, py//2 + 2*d, px//2 - 2*d, py//2 - 4*d)
  elif posicion == 'br':
      mgr.window.setGeometry(px//2 + d, py//2 + 2*d, px//2 - 2*d, py//2 - 4*d)


