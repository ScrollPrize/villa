// Constraint-chart SLIM driver. Reuses flatboi's libigl and Pastix backend.
#include <igl/slim.h>
#include <igl/flip_avoiding_line_search.h>
#include <igl/pastix6_solver.h>
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <cstdint>
#include <limits>

using Sparse = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowIndices = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template<class T> void read(std::ifstream& in, T* p, size_t n) {
    if (!in.read(reinterpret_cast<char*>(p), n*sizeof(T))) throw std::runtime_error("Truncated SLIM input");
}
void save(const std::string& path, const Eigen::MatrixXd& uv) {
    RowMatrix row = uv;
    const std::string temporary=path+".tmp";
    std::ofstream out(temporary, std::ios::binary);
    out.write(reinterpret_cast<const char*>(row.data()), row.size()*sizeof(double));
    if (!out) throw std::runtime_error("Cannot write SLIM output");
    out.close();std::filesystem::rename(temporary,path);
}
int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error("usage: winding_slim INPUT OUTPUT [iterations=50]");
        std::ifstream in(argv[1], std::ios::binary);
        uint64_t header[6]; read(in,header,6);
        const auto [magic,n64,f64,r64,k64,d64] = std::tuple{header[0],header[1],header[2],header[3],header[4],header[5]};
        if (magic != 0x534c494d57494e44ULL || n64 > INT_MAX/2 || f64 > INT_MAX || r64 > INT_MAX || d64 > 2*n64)
            throw std::runtime_error("Invalid SLIM dimensions or format");
        int n=n64, nf=f64, nr=r64, nd=d64;
        RowMatrix vr(n,3), ur(n,2); RowIndices fr(nf,3);
        Eigen::VectorXi dofs(2*n), cr(k64), cc(k64);
        Eigen::VectorXd cv(k64), target(nr);
        read(in,vr.data(),vr.size());read(in,fr.data(),fr.size());read(in,ur.data(),ur.size());
        read(in,dofs.data(),dofs.size());read(in,cr.data(),cr.size());read(in,cc.data(),cc.size());
        read(in,cv.data(),cv.size());read(in,target.data(),target.size());
        RowMatrix intrinsic(nf,3);Eigen::VectorXd mass(nf);
        read(in,intrinsic.data(),intrinsic.size());read(in,mass.data(),mass.size());
        if (!vr.allFinite() || !ur.allFinite() || !cv.allFinite() || !target.allFinite() ||
            fr.minCoeff()<0 || fr.maxCoeff()>=n || dofs.minCoeff()<0 || dofs.maxCoeff()>=nd)
            throw std::runtime_error("Invalid SLIM geometry or constraints");
        Eigen::MatrixXd vertices=vr, uv=ur;
        Eigen::MatrixXi faces=fr;
        for(int i=0;i<nf;++i) {
            Eigen::Vector2d a=uv.row(faces(i,1))-uv.row(faces(i,0));
            Eigen::Vector2d b=uv.row(faces(i,2))-uv.row(faces(i,0));
            if (a.x()*b.y()-a.y()*b.x()<=1e-12) throw std::runtime_error("SLIM requires a nonflipped initialization");
        }
        std::vector<Triplet> p, c;
        for(int i=0;i<2*n;++i)p.emplace_back(i,dofs(i),1.);
        for(int i=0;i<cv.size();++i) {
            if(cr(i)<0 || cr(i)>=nr || cc(i)<0 || cc(i)>=2*n)throw std::runtime_error("Invalid constraint index");
            c.emplace_back(cr(i),cc(i),cv(i));
        }
        Sparse P(2*n,nd), C(nr,2*n);
        P.setFromTriplets(p.begin(),p.end()); C.setFromTriplets(c.begin(),c.end());
        Sparse reducedC=C*P, penalty=reducedC.transpose()*reducedC;
        Eigen::VectorXd penaltyRhs=reducedC.transpose()*target;
        p.clear();c.clear();p.shrink_to_fit();c.shrink_to_fit();
        igl::SLIMData s;
        Eigen::MatrixXd seed(n,3);seed.leftCols(2)=uv;seed.col(2).setZero();
        igl::slim_precompute(seed,faces,uv,s,igl::MappingEnergyType::SYMMETRIC_DIRICHLET,
                             Eigen::VectorXi(),Eigen::MatrixXd(0,2),0.);
        // SLIM's intrinsic gradient and face weights come from the original
        // fine-grid surface. Coarse 3D chords across a bend or missing data
        // are not measurements of the sheet's material metric.
        std::vector<Triplet> dx,dy;
        for(int i=0;i<nf;++i) {
            double l=intrinsic(i,0),a=intrinsic(i,1),b=intrinsic(i,2);
            if(!(l>0 && b>0 && mass(i)>=0) || !std::isfinite(l+a+b+mass(i)))
                throw std::runtime_error("Invalid intrinsic triangle metric");
            dx.emplace_back(i,faces(i,0),-1/l);dx.emplace_back(i,faces(i,1),1/l);
            dy.emplace_back(i,faces(i,0),(a/l-1)/b);dy.emplace_back(i,faces(i,1),-a/(l*b));dy.emplace_back(i,faces(i,2),1/b);
        }
        s.Dx.setZero();s.Dy.setZero();s.Dx.setFromTriplets(dx.begin(),dx.end());s.Dy.setFromTriplets(dy.begin(),dy.end());
        s.M=mass;s.mesh_area=mass.sum()/2.;
        for(int i=0;i<4;++i)s.WGL_M.segment(i*nf,nf)=mass;
        dx.clear();dy.clear();dx.shrink_to_fit();dy.shrink_to_fit();
        std::function<double(Eigen::MatrixXd&)> energy = [&](Eigen::MatrixXd& value) {
            Eigen::Map<const Eigen::VectorXd> vector(value.data(),value.size());
            return igl::slim::compute_energy(s,value)+(C*vector-target).squaredNorm();
        };
        // Face weights include weak coordinate continuation across holes.
        // Every positive-weight triangle participates in flip avoidance.
        Eigen::MatrixXi barrier((mass.array()>0).count(),3);
        for(int i=0,j=0;i<nf;++i)if(mass(i)>0)barrier.row(j++)=faces.row(i);
        double e=energy(s.V_o);
        igl::Pastix6LLT solver;
        const int iterations=argc>3?std::stoi(argv[3]):50;
        std::cout<<"SLIM/Pastix: "<<n<<" vertices, "<<nf<<" triangles, "<<nd<<" degrees of freedom, "<<nr<<" constraints"<<std::endl;
        for(int iteration=0;iteration<iterations;++iteration) {
            const auto start=std::chrono::steady_clock::now();
            igl::slim::update_weights_and_closest_rotations(s,s.V_o);
            Sparse L;igl::slim::build_linear_system(s,L);
            Sparse reduced=P.transpose()*L*P+penalty;reduced.makeCompressed();
            Eigen::VectorXd rhs=P.transpose()*s.rhs+penaltyRhs;
            if(iteration==0)solver.analyzePattern(reduced);
            solver.factorize(reduced);
            Eigen::VectorXd result=P*solver.solve(rhs);
            Eigen::Map<const Eigen::MatrixXd> destination(result.data(),n,2);
            double previous=e;
            Eigen::MatrixXd direction=destination-s.V_o;
            double minStep=std::numeric_limits<double>::infinity();int limiting=-1;
            for(int i=0;i<barrier.rows();++i) {
                double step=igl::flip_avoiding::get_min_pos_root_2D(s.V_o,barrier,direction,i);
                if(step<minStep){minStep=step;limiting=i;}
            }
            std::cout<<"Flip-safe step "<<minStep<<" face "<<limiting<<std::endl;
            e=igl::flip_avoiding_line_search(barrier,s.V_o,destination,energy,e);
            if(!std::isfinite(e)||!s.V_o.allFinite())throw std::runtime_error("Nonfinite SLIM result");
            save(argv[2],s.V_o);
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            std::cout<<"Iteration "<<iteration+1<<" energy "<<e<<" relative improvement "<<(previous-e)/previous<<" seconds "<<seconds<<std::endl;
            if(std::filesystem::exists(std::string(argv[2])+".stop"))break;
            if(iteration>=5 && (previous-e)/previous < 1e-7)break;
        }
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
