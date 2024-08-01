#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include<fstream>
#include <exception>
#include <vector>
#include <math.h>

template <class T>
void printVector(const std::vector<T>& v)
{
    for (size_t i = 0; i < v.size(); i++)
        std::cout << v[i] << '\t';
    std::cout << '\n';
}

template <class T>
void endswap(T* objp)
{
    unsigned char* memp = (unsigned char*)(objp);
    std::reverse(memp, memp + sizeof(T));
}

class pictureReader
{
public:
    pictureReader(const char* fileName)
    {
        fin.open(fileName, std::ios::binary);
        if (!fin.is_open())
            throw std::exception("file error");

        fin.read((char*)&magic, 4);
        fin.read((char*)&imagesNums, 4);
        fin.read((char*)&rows, 4);
        fin.read((char*)&columns, 4);

        endswap(&magic);
        endswap(&imagesNums);
        endswap(&rows);
        endswap(&columns);
        readed = 0;
    }
    ~pictureReader()
    {
        fin.close();
    }

    std::vector<int> read()
    {
        if(readed == imagesNums)
            throw std::exception("file end");


        std::vector<int> result(rows * columns);
        unsigned char num;
        for (int i = 0; i < rows * columns; i++)
        {
            fin.read((char*)&num, 1);
            result[i] = (int)num;
        }
        readed++;
        return result;
    }
    int getRowSize()
    {
        return rows;
    }
    int getColumnSize()
    {
        return columns;
    }
    int getImageNums()
    {
        return imagesNums;
    }

private:
    std::ifstream fin;
    int magic, imagesNums, rows, columns;
    int readed;
};
class lableReader
{
public:
    lableReader(const char* fileName)
    {
        fin.open(fileName, std::ios::binary);
        if (!fin.is_open())
            throw std::exception("file error");

        fin.read((char*)&magic, 4);
        fin.read((char*)&itemsNums, 4);

        endswap(&magic);
        endswap(&itemsNums);
        readed = 0;
    }
    ~lableReader()
    {
        fin.close();
    }

    int read()
    {
        if (readed == itemsNums)
            throw std::exception("file end");
        readed++;
        
        unsigned char num;
        fin.read((char*)&num, 1);
        return (int)num;
    }
    int getItemsNums()
    {
        return itemsNums;
    }
private:
    std::ifstream fin;
    int magic, itemsNums;
    int readed;
};

enum activationFunc
{
    Sigmoid,
    Tanh,
    ReLU
};

class Matrix
{
public:
    Matrix(int n = 1, int m = 1)
    {
        data.resize(n);
        for (int i = 0; i < n; i++)
            data[i].resize(m);
    }
    Matrix(const Matrix& m)
    {
        data = m.data;
    }
    ~Matrix(){}
    const Matrix& operator =(const Matrix& m)
    {
        data = m.data;
        return *this;
    }
    
    void randInit()
    {
        for (int i = 0; i < data.size(); i++)
            for (int j = 0; j < data[i].size(); j++)
                data[i][j] = (double)rand() / (double)RAND_MAX - 0.5;
    }
    void resize(int n = 1, int m = 1)
    {
        data.resize(n);
        for (int i = 0; i < n; i++)
            data[i].resize(m);
    }
    void print()
    {
        for (int i = 0; i < data.size(); i++)
        {
            for (int j = 0; j < data[i].size(); j++)
                std::cout << data[i][j] << '\t';
            std::cout << '\n';
        }
    }
    void write(std::ofstream& fout)
    {
        if (!fout.is_open())
            throw std::exception("file error");
        fout << data.size() << ' ' << data[0].size();
        fout << '\n';
        for (int i = 0; i < data.size(); i++)
        {
            for (int j = 0; j < data[i].size(); j++)
                fout << data[i][j] << ' ';
            fout << '\n';
        }
        fout << '\n';
    }
    void read(std::ifstream& fin)
    {
        if (!fin.is_open())
            throw std::exception("file error");
        int n, m;
        fin >> n >> m;
        data.resize(n);
        for (int i = 0; i < n; i++)
            data[i].resize(m);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
                fin >> data[i][j];
        }
    }
    std::vector<double>& operator[](int x) 
    {
        return data[x];
    }
    std::vector<double> operator* (const std::vector<double>& v)
    {
        std::vector<double> result(data.size());

        for (size_t i = 0; i < data.size(); i++)
        {
            result[i] = 0.;
            for (size_t j = 0; j < data[i].size(); j++)
            {
                result[i] += data[i][j] * v[j];
            }
        }
        return result;
    }

    int getRows()
    {
        return data.size();
    }
    int getColumns()
    {
        return data[0].size();
    }
private:
    std::vector<std::vector<double>> data;
};
class perceptron
{
public:
    perceptron(activationFunc func, std::vector<int> neuronOnLayer)
    {
        setActivF(func);
        layers.resize(neuronOnLayer.size() - 1);
        for (size_t i = 0; i < neuronOnLayer.size()-1; i++)
        {
            layers[i] = Matrix(neuronOnLayer[i+1], neuronOnLayer[i]);
            layers[i].randInit();
        }
    }
    ~perceptron()
    {

    }
    int out(std::vector<double> input)
    {
        std::vector<double> layInput = input;

        for (size_t i = 0; i < layers.size(); i++)
        {
            layInput = layers[i] * layInput;
            activVector(layInput);
        }
        int ind = 0; double val = layInput[0];
        for (size_t i = 0; i < layInput.size(); i++)
        {
            if (layInput[i] > val)
            {
                ind = i;
                val = layInput[i];
            }
        }
        return ind;
    }
    void print()
    {
        for (size_t i = 0; i < layers.size(); i++)
        {
            layers[i].print();
            std::cout << '\n';
        }
    }
    void write(const char* fileName)
    {

        std::ofstream fout;
        fout.open(fileName, std::ios::binary);
        if (!fout.is_open())
            throw std::exception("file error");

        fout << funcName << '\n';
        fout << layers.size() << '\n';
        for (size_t i = 0; i < layers.size(); i++)
        {
            layers[i].write(fout);
        }
        fout.close();
    }
    void read(const char* fileName)
    {
        int f, layersSize;
        std::ifstream fin;
        fin.open(fileName, std::ios::binary);
        
        if (!fin.is_open())
            throw std::exception("file error");
        
        fin >> f;
        setActivF((activationFunc)f);
        fin >> layersSize;

        layers.resize(layersSize);
        for (int i = 0; i < layersSize; i++)
        {
            layers[i].read(fin);
        }
        fin.close();
    }
    double backPropogation(std::vector<double> input, int expected, double param)
    {
        std::vector<std::vector<double>> layerInputs(layers.size() + 1);

        layerInputs[0] = input;
        std::vector<double> layInput = input;

        for (size_t i = 0; i < layers.size(); i++)
        {
            layInput = layers[i] * layInput;
            layerInputs[i + 1] = layInput;

            activVector(layInput);
        }

        std::vector<double> error = layInput;
        error[expected] -= 1.;
        double res_er = 0;
        for (size_t i = 0; i < error.size(); i++)
            res_er += 0.5 * error[i] * error[i];

        int layer = layerInputs.size() - 1;
        std::vector<double> delta_new(layInput.size());


        for (size_t i = 0; i < delta_new.size(); i++)
            delta_new[i] = error[i] * dactivFunc(layerInputs[layer][i]);

        for (int i = 0; i < layers[layer - 1].getRows(); i++)
            for (int j = 0; j < layers[layer - 1].getColumns(); j++)
                layers[layer - 1][i][j] -= param * delta_new[i] * activFunc(layerInputs[layer-1][j]);

        std::vector<double> delta_old;

        for (layer--; layer > 0; layer--)
        {
            delta_old = delta_new;
            delta_new.resize(layerInputs[layer].size());

            for (size_t i = 0; i < delta_new.size(); i++)
            {
                delta_new[i] = 0;
                for (size_t j = 0; j < delta_old.size(); j++)
                {
                    delta_new[i] += delta_old[j] * layers[layer][j][i];
                }
                delta_new[i] *= dactivFunc(layerInputs[layer][i]);
            }

            for (int i = 0; i < layers[layer - 1].getRows(); i++)
            {
                for (int j = 0; j < layers[layer - 1].getColumns(); j++)
                {
                    layers[layer - 1][i][j] -= param * delta_new[i] * activFunc(layerInputs[layer - 1][j]);
                }
            }
        }
        return res_er;
    }
private:
    double (*activFunc) (double x);
    double (*dactivFunc) (double x);
    
    std::vector<Matrix> layers;
    activationFunc funcName;
    void activVector(std::vector<double>& v)
    {
        for (size_t i = 0; i < v.size(); i++)
            v[i] = activFunc(v[i]);
    }
    void setActivF(activationFunc func)
    {
        if (func == Sigmoid)
        {
            activFunc = [](double x) {return 1. / (1. + exp(-x)); };
            dactivFunc = [](double x) {
                if (x < -8. || x > 8.)
                    return 0.;
                return 1. / (1. + exp(-x)) * (1. - 1. / (1. + exp(-x)));
                };
            funcName = Sigmoid;
        }
        else if (func == Tanh)
        {
            activFunc = [](double x) {return 2. / (1. + exp(-2. * x)) - 1.; };
            dactivFunc = [](double x) {return 4. / pow(exp(x) + exp(-x), 2.); };
            funcName = Tanh;
        }
        else //if (func == ReLU)
        {
            activFunc = [](double x) {
                if (x > 0.)
                    return x;
                return 0.;
                };
            dactivFunc = [](double x) {
                if (x > 0.)
                    return 1.;
                return 0.;
                };
            funcName = ReLU;
        }
    }
};

template<class T, class K>
std::vector<T> convert(const std::vector<K>& v)
{
    std::vector<T> res(v.size());
    for (size_t i = 0; i < res.size(); i++)
        res[i] = static_cast<T> (v[i]);
    return res;
}

int main()
{
    srand(time(NULL));

    pictureReader imageReader("train-images.idx3-ubyte");
    lableReader labelReader("train-labels.idx1-ubyte");
    
    perceptron p(Sigmoid, { imageReader.getColumnSize() * imageReader.getRowSize(), 4000, 2000, 400, 10});
    
    int iterations = imageReader.getImageNums();
   for (int i = 0; i < iterations; i++)
    {
        std::vector<double> input = convert<double, int>(imageReader.read());
        std::cout << i+1 << "/" << iterations << " error : " << p.backPropogation(input, labelReader.read(), 0.0001) << '\n';
    }


    pictureReader imageReaderTest("t10k-images.idx3-ubyte");
    lableReader labelReaderTest("t10k-labels.idx1-ubyte");

    int error = 0;
    iterations = imageReaderTest.getImageNums();
    for (int i = 0; i < 1000; i++)
    {
        std::vector<double> input = convert<double, int>(imageReaderTest.read());
        int need = labelReaderTest.read(), get = p.out(input);
        std::cout << get << " -> " << need << '\n';
        if (need != get)
            error++;
    }
    std::cout << "\nERROR : " << error;

    p.write("net.txt");
}
