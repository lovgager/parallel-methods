#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <iterator>
#include <ctime>
#include <omp.h>

//генерация портрета матрицы смежности
void generate(int Nx, int Ny, int K3, int K4, 
        std::vector<int> &IA, std::vector<int> &JA, int T) {

    int N = Nx * Ny; //число клеток декартовой решётки
    int Nelem = N + N / (K3 + K4) * K3 + (N % (K3 + K4) < K3 ? N % (K3 + K4) : K3);
    //Nelem - число элементов сетки
    int Nnode = (Nx + 1) * (Ny + 1); //число узлов
    std::vector<int> EIA(Nelem + 1), EJA; //массивы CSR для матрицы инцидентности "элементы-узлы" (EN)
    std::vector<std::vector<int>> EJAloc(T); //для каждой нити будет свой EJA

    #pragma omp parallel for
    for (int i = 0; i < Nelem; ++i) { //цикл по элементам
        int t = omp_get_thread_num();
        int rem = i % (2 * K3 + K4);
        int cell = i / (2 * K3 + K4) * (K3 + K4) + (rem > 2 * K3 ? rem - K3 : rem / 2); 
        //cell - номер декартовой (прямоугольной) клетки, в которой лежит элемент
        int j = cell + cell / Ny; //номер узла в верхнем левом углу декартовой клетки

        //направление обхода узлов данного элемента - против часовой стрелки
        if (rem >= 2 * K3) { //если элемент четырёхугольный
            EIA[i + 1] = 4;
            EJAloc[t].push_back(j);
            EJAloc[t].push_back(j + Ny + 1);
            EJAloc[t].push_back(j + Ny + 2);
            EJAloc[t].push_back(j + 1);
        }
        else {
            EIA[i + 1] = 3;
            if (rem % 2) { //если элемент треугольный (правый верхний)
                EJAloc[t].push_back(j);
                EJAloc[t].push_back(j + Ny + 2);
                EJAloc[t].push_back(j + 1);
            }
            else { //если левый нижний
                EJAloc[t].push_back(j);
                EJAloc[t].push_back(j + Ny + 1);
                EJAloc[t].push_back(j + Ny + 2);
            }
        }
    }
    for (int i = 0; i < Nelem; ++i) { //кумулятивное суммирование в EIA, получим смещения
        EIA[i + 1] += EIA[i];
    }
    for (int t = 0; t < T; ++t) { //объединяем все локальные EJAloc в один EJA
        std::move(EJAloc[t].begin(), EJAloc[t].end(), std::back_inserter(EJA));
    }

    std::vector<int> NIA(Nnode + 1), NJA; //матрица "узлы-элементы" (NE) в формате CSR
    //транспонирование матрицы EN
    #pragma omp parallel for
    for (int i = 0; i < Nelem; ++i) {
        for (int k = EIA[i]; k < EIA[i + 1]; ++k) {
            int j = EJA[k];
            NIA[j + 1]++;
        }
    }
    for (int i = 1; i <= Nnode; ++i) {
        NIA[i] += NIA[i - 1];
    }
    NJA.resize(NIA[Nnode], -1);
    #pragma omp parallel for
    for (int i = 0; i < Nelem; ++i) {
        for (int k = EIA[i]; k < EIA[i + 1]; ++k) {
            int j = EJA[k];
            int b = NIA[j];
            int e = NIA[j + 1] - 1;
            NJA[e]++;
            NJA[b + NJA[e]] = i;
        }
    }

    //составляем матрицу смежности узлов по рёбрам (Node-Edge-Node)
    IA.resize(Nnode + 1, 0);
    std::vector<std::vector<int>> JAloc(T);
    #pragma omp parallel for
    for (int node = 0; node < Nnode; ++node) { //цикл по узлам
        std::set<int> adjNodes;
        adjNodes.insert(node);
        for (int k = NIA[node]; k < NIA[node + 1]; ++k) {
            int elem = NJA[k]; //номер элемента, инцидентного данному узлу
            int begin = EIA[elem];
            int end = EIA[elem + 1] - 1;
            for (int kk = begin; kk <= end; ++kk) {
                if (EJA[kk] == node) {
                    if (kk == begin) {
                        adjNodes.insert(EJA[end]);
                        adjNodes.insert(EJA[begin + 1]);
                    }
                    else if (kk == end) {
                        adjNodes.insert(EJA[end - 1]);
                        adjNodes.insert(EJA[begin]);
                    }
                    else {
                        adjNodes.insert(EJA[kk - 1]);
                        adjNodes.insert(EJA[kk + 1]);
                    }
                }
            }
        }
        IA[node + 1] = adjNodes.size();
        int t = omp_get_thread_num();
        for (auto i = adjNodes.begin(); i != adjNodes.end(); ++i) {
            JAloc[t].push_back(*i);
        }
    }
    for (int node = 0; node < Nnode; ++node) {
        IA[node + 1] += IA[node];
    }
    for (int t = 0; t < T; ++t) {
        std::move(JAloc[t].begin(), JAloc[t].end(), std::back_inserter(JA));
    }
}


//построение СЛАУ по заданному портрету матрицы
void fill(const std::vector<int>& IA, 
          const std::vector<int>& JA, 
          std::vector<double>& A,
          std::vector<double>& b, int T) {
    std::vector<std::vector<double>> Aloc(T);
    b.resize(IA.size() - 1);
    #pragma omp parallel for
    for (int i = 0; i < IA.size() - 1; ++i) {
        int t = omp_get_thread_num();
        double row_sum = 0.0;
        int diag_index;
        int runner = Aloc[t].size();
        for (int k = IA[i]; k < IA[i + 1]; ++k) {
            int j = JA[k]; //номер столбца
            double aij = cos(i * j + i + j);
            if (i == j) {
                //запоминаем позицию диагонального элемента
                diag_index = runner;
                Aloc[t].push_back(0.0); //заполним диагональный элемент позже
                ++runner;
                continue;
            }
            row_sum += abs(aij);
            Aloc[t].push_back(aij);
            ++runner;
        }
        Aloc[t][diag_index] = 1.234 * row_sum; //обеспечиваем диагональное преобладание
        b[i] = sin(i);
    }
    for (int t = 0; t < T; ++t) {
        std::move(Aloc[t].begin(), Aloc[t].end(), std::back_inserter(A));
    }
}


//линейная комбинация векторов: ax + y
void axpy(double a,
          const std::vector<double>& x,
          const std::vector<double>& y,
          std::vector<double>& res) {
    int N = x.size();
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        res[i] = a * x[i] + y[i];
    }
}


//скалярное произведение векторов
double dot(const std::vector<double>& x,
           const std::vector<double>& y) {
    int N = x.size();
    double res = 0.0;
    #pragma omp parallel for reduction(+:res)
    for (int i = 0; i < N; ++i) {
        res += x[i] * y[i];
    }
    return res;
}


//умножение CSR-матрицы на вектор
void SpMV(const std::vector<int>& IM,
          const std::vector<int>& JM,
          const std::vector<double>& M,
          const std::vector<double>& v,
          std::vector<double>& res) {
    int N = IM.size() - 1;
    res.resize(N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        res[i] = 0.0;
        for (int k = IM[i]; k < IM[i + 1]; ++k) {
            int j = JM[k];
            res[i] += M[k] * v[j];
        }
    }
}


//решение СЛАУ итерационным методом
void solve(const std::vector<int>& IA,
           const std::vector<int>& JA,
           const std::vector<double>& A,
           const std::vector<double>& b,
           double eps, int maxit,
           std::vector<double>& x, //выходной вектор
           int &n, //количество выполненных итераций
           double &res //норма невязки
) {
    //параметры метода
    int N = IA.size() - 1;
    std::vector<double> r = b, z(N), p(N), q(N);
    x.resize(N);
    int k = 0;
    double alpha, beta, rho, rho_prev = -1;
    std::vector<int> IM(N + 1), JM(N);
    std::vector<double> M(N);

    //вытаскиваем главную диагональ из матрицы A
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        IM[i] = i;
        JM[i] = i;
    }
    IM[N] = N;
    for (int i = 0; i < N; ++i) {
        for (int k = IA[i]; k < IA[i + 1]; ++k) {
            int j = JA[k];
            if (i == j)
                M[i] = 1/A[k]; //сразу обращаем элементы главной диагонали
        }
    }

    //итерационный процесс
    do {
        ++k;
        SpMV(IM, JM, M, r, z);
        z = r;
        rho = dot(r, z);
        if (k == 1)
            p = z;
        else {
            beta = rho / rho_prev;
            axpy(beta, p, z, p);
        }
        SpMV(IA, JA, A, p, q);
        alpha = rho / dot(p, q);
        axpy(alpha, p, x, x);
        axpy(-alpha, q, r, r);
        rho_prev = rho;
    } while (rho > eps && k < maxit);

    n = k;
    res = sqrt(dot(r, r));
}


int main(int argc, char** argv)
{
    setlocale(LC_ALL, "Russian");
    if (argc == 1) {
        std::cout << "Формат ввода: FirstTask <Nx> <Ny> <K3> <K4> <T> [print]" << std::endl;
        std::cout << "Nx - число клеток в решётке по вертикали" << std::endl;
        std::cout << "Ny - число клеток в решётке по горизонтали" << std::endl;
        std::cout << "K3 - параметр для треугольных элементов" << std::endl;
        std::cout << "K4 - параметр для четырёхугольных элементов" << std::endl;
        std::cout << "T - число нитей OpenMP" << std::endl;
        std::cout << "print - вывод результатов в файл (0 или 1)" << std::endl;
        return 0;
    }
    if (argc < 6) {
        std::cout << "Недостаточно аргументов" << std::endl;
        return 0;
    }
    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int K3 = atoi(argv[3]);
    int K4 = atoi(argv[4]);
    int T  = atoi(argv[5]);
    bool print = (argc > 6 ? bool(atoi(argv[6])) : 0);
    if (Nx < 1 or Ny < 1 or K3 < 0 or K4 < 0 or T <= 0) {
        std::cout << "Недопустимые значения" << std::endl;
        return 0;
    }
    std::vector<int> IA, JA;
    std::vector<double> A, b, x;
    double res;
    int n;

    omp_set_num_threads(T);

    double start_time = omp_get_wtime();
    generate(Nx, Ny, K3, K4, IA, JA, T);
    double generate_time = omp_get_wtime() - start_time;

    start_time = omp_get_wtime();
    fill(IA, JA, A, b, T);
    double fill_time = omp_get_wtime() - start_time;

    start_time = omp_get_wtime();
    solve(IA, JA, A, b, 1e-10, 1000, x, n, res);
    double solve_time = omp_get_wtime() - start_time;

    std::cout << "Время выполнения (с)" << std::endl;
    std::cout << "Генерация: " << generate_time << std::endl;
    std::cout << "Заполнение: " << fill_time << std::endl;
    std::cout << "Решение: " << solve_time << std::endl;
    std::cout << "В сумме: " << generate_time + fill_time + solve_time << std::endl;

    //печать результатов в файл
    if (print) {
        std::ofstream fout("result.txt");

        fout << "Число узлов сетки: " << (Nx + 1) * (Ny + 1) << std::endl;
        fout << "Смежность узлов по рёбрам: " << std::endl;
        for (int i = 0; i < IA.size() - 1; ++i) {
            fout << i << " : ";
            for (int k = IA[i]; k < IA[i + 1]; ++k) {
                fout << JA[k] << " ";
            }
            fout << std::endl;
        }
        
        fout << std::endl << "Матрица: " << std::endl;
        for (int i = 0; i < IA.size() - 1; ++i) {
            fout << i << " : ";
            for (int k = IA[i]; k < IA[i + 1]; ++k) {
                fout << A[k] << " ";
            }
            fout << std::endl;
        }

        fout << std::endl << "Правая часть: " << std::endl;
        for (int i = 0; i < IA.size() - 1; ++i) {
            fout << b[i] << " ";
        }

        fout << std::endl << std::endl << "Вектор решения: " << std::endl;
        for (int i = 0; i < IA.size() - 1; ++i) {
            fout << x[i] << " ";
        }

        fout << std::endl << std::endl << "Невязка: " << std::endl;
        fout << res << std::endl;

        fout << std::endl << "Число итераций: " << n << std::endl;

        fout.close();
    }
}