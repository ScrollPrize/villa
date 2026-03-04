# What components to install
unset(install_components)
if(VC_INSTALL_APPS)
    set(install_components ${install_components} Programs)
endif()
if(VC_INSTALL_LIBS)
    set(install_components ${install_components} Libraries)
endif()
if(VC_INSTALL_UTILS)
    set(install_components ${install_components} Utilities)
endif()
if(VC_INSTALL_EXAMPLES)
    set(install_components ${install_components} Examples)
endif()
if(VC_INSTALL_DOCS)
    set(install_components ${install_components} Documentation)
endif()

# Install resources (e.g. README, LICENSE, etc.) if anything else is
# getting installed
if(install_components)
install(
  FILES "LICENSE" "NOTICE"
  DESTINATION "${share_install_dir}"
  COMPONENT Resources
)
endif()

# Configure Cpack
