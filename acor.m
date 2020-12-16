function [variables, sol] = acor(n, dom)
%% Descripción general de la función %%

% Esta función consta de 2 parámetros de entrada directos (deben
% introducirse en el momento de llamar a la función), que son:
%     n   : El número de variables de decisión = dimensión del modelo --> n
%     dom : Matriz nx2 en la que se indica el dominio de cada variable de
%           decisión Xi
% y de otros parámetros indirectos, que se editarán dentro del código, en
% la sección denominada "Parámetros del modelo"

% El procedimiento devuelve 2 parámetros de salida, el primero
% correspondiente al valor óptimo de las variables de decisión y el segundo
% correspondiente al valor óptimo de la función coste, que es el objetivo a
% minimizar

%% Ejemplo de uso %%

% Consideremos por ejemplo 3 variables de decisión, cuyos dominios vienen
% dados por:
%   x1 : [-1 4]
%   x2 : [5, 7]
%   x3 : [9, 11]
% El resto de parámetros de modelo serán los que vienen por defecto:
%   cost_function = @(x) sum(x.^2); % Modelo de esfera
%   q = 0.005;
%   z = 0.8;
%   max_it = 1000;
%   k = 40;
% Con esto, llamaríamos a la función de la siguiente manera:
%   [var, sol] = acor(3, [-1 4;5 7;9 11]
% Y puede probarse que la solución es la que cabe esperar:
%   >> var
%       0.0000  5.0000  9.0000
%   >> sol
%       106

%% Comprobación de parámetros %%

% En primer lugar se comprueba que se han introducido los rangos de dominio
% de todas las variables de manera adecuada
if ~isequal( size(dom) , [n, 2] )
    error('Deben introducirse unos rangos de dominio para cada variable')
end

%% Parámetros del modelo %%

cost_function = @(x) sum(x.^2); % Función coste
                                % Es la que pretendemos minimizar

% --- Parámetros de la distribución de probabilidad ---                                
q = 0.005;   % Factor de intensificación
z = 0.8;     % Ratio de evaporación de las feromonas.

max_it = 1000; % Número máximo de iteraciones del modelo                                
                                
% --- Parámetros de la matriz o fichero de soluciones ---
k = 40;  % Tamaño del archivo de soluciones --> k. Debe ser mayor que n
         % Las n primeras columnas corresponderán al nº de variables X
  
%% Configuración inicial del archivo de soluciones %%

% En el primer paso, cada una de las k hormigas construirá su propia
% solucion de manera aleatoria

data = zeros(k, n + 1); % Inicialización del archivo de soluciones (data). 
                        % Es una matriz k x (n+1). Su última columna
                        % contiene las evaluaciones de la función coste, f,
                        % sobre las soluciones que construyen las
                        % k hormigas
%    ________________________________
% s1 | s11 | s12 | ... | s1n | f(s1) | \
%    |_____|_____|_____|_____|_______|  |
% s2 | s21 | s22 | ... | s2n | f(s2) |  |
%    |_____|_____|_____|_____|_______|  |
%       .     .     .     .      .       >  k
%       .     .     .     .      .      |
%    ________________________________   |
% sk | sk1 | sk2 | ... | skn | f(sk) |  |
%    |_____|_____|_____|_____|_______| /
%
%    \______________  ______________/
%                   \/
%                  n + 1

for i = 1 : n % Bucle a las n columnas de data, para fijar cada una de
              % las soluciones iniciales
                          
    data(:, i) = unifrnd(dom(i,1), dom(i,2), 1, k);
    % unifrnd genera un número aleatorio entre las 2 cotas introducidas
    
end

for l = 1 : k % Bucle a las filas de data, para evaluar la función coste
              % sobre la solución anteriormente encontrada. El resultado lo
              % almacenamos en la última columna de la matriz data
              
    data(l, end) = cost_function(data(l, 1:end-1));

end

% --- Construcción de los pesos Gaussianos ---
data = sortrows(data, n + 1);
% En primer lugar debemos ordenar el archivo de soluciones de menor a mayor
% éxito (esto es, de menor a mayor valor de f(si), ya que hay que
% minimizar)

w = 1/(q*k*sqrt(2*pi))*exp(-0.5*(((1:k)-1)/(q^2*k^2)).^2);
% Asignamos un peso gaussiano a cada una de las filas (soluciones) de la
% matriz T de datos (ya ordenada). Estos pesos se almacenan en este array

p = w/sum(w); % Array con las probabilidades de escoger la i-ésima función
              % gaussiana
              
% --- Definimos una función para samplear      
function y = kernel_selector(P)
    % P es un array que contiene las probabilidades de elección de kernel
    % gaussiano; es decir, es una distribución de probabilidad
    
    r = rand;      % Generamos un nº aleatorio entre 0 y 1
    C = cumsum(P); % Hallamos la función de distribución acumulativa, CDF
    y = find(r <= C, 1); % Devolvemos el índice del primer elemento de C
                         % que es superior o igual al nº aleatorio generado
                         % Esto es equivalente a utilizar la inversa de C.
    
end

%% Construcción del bucle principal del algoritmo %%

% La idea es ir actualizando el archivo de soluciones T de manera iterada.
% En cada paso, las hormigas encontrarán una nueva solución eligiendo un
% elemento dentro del espacio de búsqueda, que seleccionarán siguiendo una
% distribución de probabilidad compuesta de diferentes kernel gaussianos.

for t = 1 : max_it
    
    % --- Construcción del Kernel Gausiano --
    % Un Kernel Gaussiano se parametriza con 3 arrays, que corresponden al
    % array de medias mu, al array de pesos w y al array de desviaciones
    % sigma
    % Su construcción se realiza siempre a partir de las componentes i de
    % todas las soluciones k
    
    % Definimos la matriz que lleva las medias, que es igual a la matriz de
    % soluciones (exceptuando la última columna, que lleva las funciones f)
    mu = data(:,1:end-1); % Dimensión kxn
    
    % Definimos la matriz de desviaciones estandar   
    sigma=zeros(k,n); % Inicialización de la matriz sigma
    
    for i = 1 : k
        for l = 1 : n
            D = 0; % D será la suma de las distancias entre soluciones
            for e = 1 : k
                D = D + abs( mu(e,l) - mu(i,l) );
            end
            sigma(i,l) = z/(k-1)*D; % Expresión para la componente (i,l)
        end
    end
    
    % --- Construcción de soluciones ---
    
    new_data = data;  % Construimos una nueva matriz de soluciones
    
    for new = 1 : k   % Bucle a las filas del archivo data (hormigas)
        
        for i = 1 : n % Bucle a las variables de decisión
            
            ker = kernel_selector(p); % Seleccionamos un Kernel gaussiano
                                      % de acuerdo al array de
                                      % probabilidades p. ker será un
                                      % índice que marca el número de
                                      % kernel que vamos a usar
            
            new_data(new,i) = mu(i,ker) + sigma(i,ker)*randn;
            % Generamos una variable gaussiana con media mu y desviación
            % estandar sigma. Como se ve, cada hormiga emplea únicamente
            % las variables de decisión i
          
           while new_data(new,i) < dom(i,1) || new_data(new,i) > dom(i,2)
               new_data(new,i) = mu(ker,i) + sigma(ker,i)*randn;
               % Esto lo que hace es descartar soluciones que se salgan del
               % dominio de las variables de búsqueda
           end
           
        end
        
        % Re-calculamos la función de coste, evaluando sobre la nueva tabla
        % de datos
        new_data(new, end) = cost_function(new_data(new, 1:end-1));
 
    end % Esto finaliza la construcción de la nueva tabla de datos
    
    % Juntamos las 2 listas de datos y reordenamos según la función
    % coste que acabamos de evaluar
    data = [data; new_data];
    data = sortrows(data, n + 1);
    
    % Nos quedamos únicamente con las k mejores soluciones
    data = data(1:k, :);
    sol  = data(1, end); % Apuntamos el valor de la mejor solución
    
end

variables = data(1,1:end - 1); % Variable de salida

end