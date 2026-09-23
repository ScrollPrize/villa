// Offscreen replay of the fiber review Auto link button on a saved review
// without decisions: auto link the selected fiber, walk the auto-rejected
// pairs, save, and reopen. Build with prepare_fiber_review_benchmark.py
// --source probe_fiber_review_auto_link.cpp (links the built VC3D objects).
#include "ConstraintGraphWorkspace.hpp"
#include "CState.hpp"
#include "vc/core/types/Volume.hpp"
#include <QApplication>
#include <QElapsedTimer>
#include <QListWidget>
#include <QPushButton>
#include <QThread>
#include <fstream>
#include <iostream>
namespace cg=vc::constraints;
namespace {
void settle(ConstraintGraphWorkspace& w,int minimumMs,int timeoutMs) {
    QElapsedTimer timer;timer.start();
    while(w.busy()||timer.elapsed()<minimumMs){QCoreApplication::processEvents();QThread::msleep(2);if(timer.elapsed()>timeoutMs)throw std::runtime_error("timeout while waiting for the workspace");}
}
}
int main(int argc,char** argv) {
    if(argc!=4){std::cerr<<"Usage: probe_fiber_review_auto_link REVIEW.json VOLUME.zarr REPORT.json\n";return 2;}
    QApplication app(argc,argv);app.setOrganizationName("VCReviewAutoLinkProbe");app.setApplicationName("VCReviewAutoLinkProbe");
    try {
        CState main;auto volume=Volume::New(argv[2]);main.setCurrentVolume(volume);
        ConstraintGraphWorkspace workspace(&main);workspace.resize(1600,1000);workspace.show();workspace.synchronizeVolume(volume);
        workspace.openSession(argv[1]);settle(workspace,1000,600000);
        auto* autoLink=workspace.findChild<QPushButton*>("constraintGraphAutoLinkFiber");
        auto* next=workspace.findChild<QPushButton*>("constraintGraphNextAutoRejected");
        auto* decisions=workspace.findChild<QListWidget*>("constraintGraphReviewDecisions");
        if(!autoLink||!next||!decisions)throw std::runtime_error("auto-link controls missing");
        // Discovery of the selected fiber may still be pending; wait for the proposal search too.
        settle(workspace,2000,600000);
        if(!autoLink->isEnabled())throw std::runtime_error("Auto link fiber is disabled after loading");
        nlohmann::json report;
        auto count=[&](const QString& prefix){int n=0;for(int i=0;i<decisions->count();++i)n+=decisions->item(i)->text().startsWith(prefix);return n;};
        // Decisions present right after layout (a compact workspace fits its accepted links here).
        report["loaded_auto_accepted"]=count("Auto-accepted");report["loaded_auto_rejected"]=count("Auto-rejected");report["loaded_auto_skipped"]=count("Auto-skipped");
        std::cout<<"after load: auto-accepted="<<report["loaded_auto_accepted"]<<" auto-rejected="<<report["loaded_auto_rejected"]<<" auto-skipped="<<report["loaded_auto_skipped"]<<std::endl;
        QElapsedTimer timer;timer.start();
        autoLink->click();settle(workspace,500,1800000);
        report["auto_link_seconds"]=timer.elapsed()/1000.;
        report["auto_accepted"]=count("Auto-accepted");report["auto_rejected"]=count("Auto-rejected");report["auto_skipped"]=count("Auto-skipped");
        report["decisions"]=decisions->count();
        std::cout<<"auto-accepted="<<report["auto_accepted"]<<" auto-rejected="<<report["auto_rejected"]<<" auto-skipped="<<report["auto_skipped"]<<" seconds="<<report["auto_link_seconds"]<<std::endl;
        for(int i=0;i<std::min(6,decisions->count());++i)std::cout<<"  "<<decisions->item(i)->text().toStdString().substr(0,60)<<" | "<<decisions->item(i)->toolTip().toStdString()<<std::endl;
        if(report["auto_accepted"].get<int>()==0)throw std::runtime_error("nothing was auto-linked");
        // Walk the auto-rejected pairs: each click reopens one for review.
        int walked=0;
        while(next->isEnabled()&&walked<3){const int before=count("Auto-rejected");next->click();settle(workspace,300,600000);if(count("Auto-rejected")!=before-1)throw std::runtime_error("Next auto-rejected did not reopen exactly one pair");++walked;}
        report["walked_auto_rejected"]=walked;
        // Saving goes through the toolbar; persistence of the automatic mark is covered by test_constraint_graph.
        std::cout<<"walked="<<walked<<std::endl;
        std::ofstream(argv[3])<<report.dump(2);
        return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
