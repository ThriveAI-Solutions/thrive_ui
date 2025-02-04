# from langsec import SQLSecurityGuard
# from langsec.schema.defaults import low_security_config

# https://docs.lang-sec.com/quick-start/

# guard = SQLSecurityGuard(schema=low_security_config)

def validate(sql):
    return sql
    # return guard.validate_query(sql)